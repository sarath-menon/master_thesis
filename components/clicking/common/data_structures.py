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

class ObjectValidity(BaseModel):
    is_valid: bool = Field(default=True)
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