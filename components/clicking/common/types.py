from typing import List, Optional
from typing import TypedDict, List, Optional, NamedTuple
from PIL import Image
from pydantic import BaseModel, Field
from enum import Enum
from clicking.common.mask import SegmentationMask
from clicking.common.bbox import BoundingBox
from pydantic import ConfigDict

class ObjectCategory(str, Enum):
    GAME_ASSET = "Game Asset"
    INFORMATION_DISPLAY = "Information Display"
    NPC = "Non-playable Character"

# Add this new dictionary for category colors
CATEGORY_COLOR_MAP = {
    ObjectCategory.GAME_ASSET: (1, 0, 0),        # Red
    ObjectCategory.INFORMATION_DISPLAY: (0, 1, 0),  # Green
    ObjectCategory.NPC: (0, 0, 1)                # Blue
}

class ImageObject(BaseModel):
    name: str
    description: str
    category: ObjectCategory
    bbox: Optional[BoundingBox] = None
    mask: Optional[SegmentationMask] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

class ClickingImage(BaseModel):
    image: Image.Image
    id: str
    true_objects: List[ImageObject] = Field(default_factory=list)
    predicted_objects: List[ImageObject] = Field(default_factory=list)
    model_config = ConfigDict(arbitrary_types_allowed=True)


