from typing import List, Optional
from typing import TypedDict, List, Optional, NamedTuple
from PIL import Image

class ImageObject(TypedDict):
    name: str
    category: str
    description: str

class SinglePromptResponse(TypedDict):
    objects: List[ImageObject]

class ImageWithDescriptions(NamedTuple):
    image: Image.Image
    id: str
    object_name: str
    description: Optional[SinglePromptResponse] = None