from typing import List, Dict, Optional, TypedDict, Union, NamedTuple
from enum import Enum, auto
from PIL import Image
from clicking.common.data_structures import *
from pydantic import BaseModel
from typing import Type

class ObjectsResponse(BaseModel):
    objects: list[ImageObject]

class UIResponse(BaseModel):
    objects: list[UIElement]

class PromptMode(Enum):
    OBJECTS_LIST_TO_DESCRIPTIONS = ("OBJECTS_LIST_TO_DESCRIPTIONS", ObjectsResponse)
    IMAGE_TO_OBJECT_DESCRIPTIONS = ("IMAGE_TO_OBJECT_DESCRIPTIONS", ObjectsResponse)
    IMAGE_TO_OBJECTS_LIST = ("IMAGE_TO_OBJECTS_LIST", ObjectsResponse)
    IMAGE_TO_UI_ELEMENTS = ("IMAGE_TO_UI_ELEMENTS", UIResponse)

    def __init__(self, value: str, response_type: Type):
        self._value_ = value
        self.response_type = response_type

class ProcessedPrompts(NamedTuple):
    samples: List[ClickingImage]
 
class TemplateValues(TypedDict, total=False):
    input_description: str
    word_limit: str
    description_length: int

class SingleObjectsResponse(TypedDict):
    prompt: str
    mode: PromptMode
    template_values: TemplateValues