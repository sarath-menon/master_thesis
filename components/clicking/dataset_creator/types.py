from typing import List, NamedTuple
from PIL import Image

class DataSample(NamedTuple):
    image: Image.Image
    class_label: str

class DatasetSample(NamedTuple):
    samples: List[DataSample]