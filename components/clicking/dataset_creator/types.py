from typing import List, NamedTuple
from PIL import Image


class DatasetSample(NamedTuple):
    images: List[Image.Image]
    object_names: List[str]