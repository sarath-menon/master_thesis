from typing import List, Dict, Optional, TypedDict, Union, NamedTuple
from enum import Enum, auto
from PIL import Image

class PromptMode(Enum):
    OBJECTS_LIST_TO_DESCRIPTIONS = "OBJECTS_LIST_TO_DESCRIPTIONS"
    IMAGE_TO_OBJECT_DESCRIPTIONS = "IMAGE_TO_OBJECT_DESCRIPTIONS"
    IMAGE_TO_OBJECTS_LIST = "IMAGE_TO_OBJECTS_LIST"

class ObjectDescription(TypedDict):
    name: str
    category: str
    description: str

class SinglePromptResponse(TypedDict):
    objects: List[ObjectDescription]

class ProcessedSample(NamedTuple):
    image: Image.Image
    image_id: str
    object_name: str
    description: SinglePromptResponse

class ProcessedPrompts(NamedTuple):
    samples: List[ProcessedSample]

class TemplateValues(TypedDict, total=False):
    input_description: str
    word_limit: str
    description_length: int