from typing import List, Optional, Callable
from typing import TypedDict, List, Optional, NamedTuple
from PIL import Image
from pydantic import BaseModel, Field, field_validator
from enum import Enum
from clicking.common.mask import SegmentationMask
from clicking.common.bbox import BoundingBox
from pydantic import ConfigDict
import uuid
from typing import Literal
from dataclasses import dataclass, field
import random
import hashlib
import pickle

class ObjectCategory(str, Enum):
    GAME_ASSET = "Game Asset"
    INFORMATION_DISPLAY = "Information Display"
    NPC = "Non-playable Character"
    OTHER = "Other"

    @field_validator('category', mode='before')
    def validate_category(cls, value):
        try:
            return ObjectCategory(value)
        except ValueError:
            print(f"Invalid object category: {value}, defaulting to OTHER")
            return ObjectCategory.OTHER  # Default fallback value

# Add this new dictionary for category colors
CATEGORY_COLOR_MAP = {
    ObjectCategory.GAME_ASSET: 'yellow',       
    ObjectCategory.INFORMATION_DISPLAY: 'green',  
    ObjectCategory.NPC: 'blue',                
}

class Validity(BaseModel):
    is_valid: bool = Field(default=True)
    reason: Optional[str] = None


class ValidityStatus(Enum):
    UNKNOWN = "unknown"
    VALID = "valid"
    INVALID = "invalid"

class ObjectValidity(BaseModel):
    status: ValidityStatus = Field(default=ValidityStatus.UNKNOWN)
    accuracy: Literal["true", "false"] = Field(default="true")
    visibility: Literal["fully visible", "partially visible", "hidden"] = Field(default="fully visible")
    reason: Optional[str] = None

class ImageObject(BaseModel):
    id: str = Field(default_factory=uuid.uuid4)
    name: str
    description: Optional[str] = None
    category: Optional[ObjectCategory] = None
    bbox: Optional[BoundingBox] = None
    mask: Optional[SegmentationMask] = None
    significance: Optional[str] = None
    validity: ObjectValidity = Field(default=ObjectValidity())

    model_config = ConfigDict(arbitrary_types_allowed=True)

class ClickingImage(BaseModel):
    image: Image.Image
    id: str
    annotated_objects: List[ImageObject] = Field(default_factory=list)
    predicted_objects: List[ImageObject] = Field(default_factory=list)
    model_config = ConfigDict(arbitrary_types_allowed=True)

class ModuleMode(NamedTuple):
    name: str
    handler: Callable[[object], str]

class ObjectImageDict(BaseModel):
    image_id: str
    object: ImageObject

    model_config = ConfigDict(arbitrary_types_allowed=True)


@dataclass
class PipelineState:
    images: List[ClickingImage] = field(default_factory=list)

    def __hash__(self):
        # Create a hash based on the content of PipelineState
        return hash(pickle.dumps(self))

    def __eq__(self, other):
        if not isinstance(other, PipelineState):
            return False
        return self.images == other.images

    def filter_by_object_category(self, category: ObjectCategory):
        for clicking_image in self.images:
            clicking_image.predicted_objects = [
                obj for obj in clicking_image.predicted_objects
                if obj.category == category
            ]
        return self

    def get_image_by_id(self, image_id: str) -> ClickingImage:
        for image in self.images:
            if image.id == image_id:
                return image
        raise ValueError(f"Image with id {image_id} not found")

    def filter_by_ids(self, image_ids: Optional[List[int]] = None, sample_size: Optional[int] = None):
        if image_ids is not None and sample_size is not None:
            raise ValueError("Cannot specify both image_ids and sample_size. Choose one filtering method.")
        
        if image_ids is not None:
            image_ids = [str(id) for id in image_ids]
            self.images = [img for img in self.images if img.id in image_ids]
        elif sample_size is not None:
            if sample_size > len(self.images):
                raise ValueError(f"Sample size {sample_size} is larger than the number of available images {len(self.images)}")
            self.images = random.sample(self.images, sample_size)
        
        return self

    
    def get_all_predicted_objects(self) -> dict[str, ObjectImageDict]:
        all_objects = {}
        for image in self.images:
            for obj in image.predicted_objects:
                all_objects[str(obj.id)] = ObjectImageDict(image_id=str(image.id), object=obj)
        return all_objects