from typing import List, Optional
from typing import TypedDict, List, Optional, NamedTuple
from PIL import Image

class ObjectDescription(TypedDict):
    name: str
    category: str
    description: str

class SinglePromptResponse(TypedDict):
    objects: List[ObjectDescription]

class ImageWithDescriptions(NamedTuple):
    image: Image.Image
    image_id: str
    object_name: str
    description: Optional[SinglePromptResponse] = None