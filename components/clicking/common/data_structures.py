from typing import List, Optional
from typing import TypedDict, List, Optional, NamedTuple
from PIL import Image
from pydantic import BaseModel, Field, field_validator
from enum import Enum
from clicking.common.mask import SegmentationMask
from clicking.common.bbox import BoundingBox
from pydantic import ConfigDict
import uuid

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
    ObjectCategory.GAME_ASSET: (1, 0, 0),        # Red
    ObjectCategory.INFORMATION_DISPLAY: (0, 1, 0),  # Green
    ObjectCategory.NPC: (0, 0, 1)                # Blue
}

class Validity(BaseModel):
    is_valid: bool = Field(default=True)
    reason: Optional[str] = None

class ImageObject(BaseModel):
    id: str = Field(default_factory=uuid.uuid4)
    name: str
    description: Optional[str] = None
    category: Optional[ObjectCategory] = None
    bbox: Optional[BoundingBox] = None
    mask: Optional[SegmentationMask] = None
    validity: Validity = Field(default=Validity())

    model_config = ConfigDict(arbitrary_types_allowed=True)

class ClickingImage(BaseModel):
    image: Image.Image
    id: str
    annotated_objects: List[ImageObject] = Field(default_factory=list)
    predicted_objects: List[ImageObject] = Field(default_factory=list)
    model_config = ConfigDict(arbitrary_types_allowed=True)


