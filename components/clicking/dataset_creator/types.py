from typing import List, NamedTuple
from PIL import Image

class ImageSample(NamedTuple):
    image: Image.Image
    object_name: str
    id: int

class DatasetSample(NamedTuple):
    images: List[ImageSample]