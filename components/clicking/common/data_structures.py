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
from dataclasses import dataclass, field
import random
import hashlib
import pickle

class TaskType(str, Enum):
    LOCALIZATION_WITH_TEXT = "LOCALIZATION_WITH_TEXT"
    LOCALIZATION_WITH_TEXT_GROUNDED = "LOCALIZATION_WITH_TEXT_GROUNDED"
    LOCALIZATION_WITH_TEXT_OPEN_VOCAB = "LOCALIZATION_WITH_TEXT_OPEN_VOCAB"
    SEGMENTATION_WITH_TEXT = "SEGMENTATION_WITH_TEXT"
    SEGMENTATION_WITH_CLICKPOINT = "SEGMENTATION_WITH_CLICKPOINT"
    SEGMENTATION_WITH_BBOX = "SEGMENTATION_WITH_BBOX"
    SEGMENTATION_WITH_CLICKPOINT_AND_BBOX = "SEGMENTATION_WITH_CLICKPOINT_AND_BBOX"
    CAPTIONING = "CAPTIONING"
    SEGMENTATION_AUTO_ANNOTATION = "SEGMENTATION_AUTO_ANNOTATION"
    OCR = "OCR"
    CLICKPOINT_WITH_TEXT = "CLICKPOINT_WITH_TEXT"

class UIElement(BaseModel):
    name: Optional[str] = None
    icon: Optional[str] = None
    interaction: str
    category: str
    shape: str
    color: str
    location: str
    bbox: Optional[BoundingBox] = None

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

class Validity(BaseModel):
    is_valid: bool = Field(default=True)
    reason: Optional[str] = None


class ValidityStatus(Enum):
    UNKNOWN = "unknown"
    VALID = "valid"
    INVALID = "invalid"


class ObjectValidity(BaseModel):
    status: ValidityStatus = Field(default=ValidityStatus.UNKNOWN)
    accuracy: Optional[Literal["true", "false"]] = Field(default=None)
    visibility: Optional[Literal["fully visible", "partially visible", "hidden"]] = Field(default=None)
    reason: Optional[str] = None

class ClickPoint(BaseModel):
    x: Optional[float] = Field(None, description="X-coordinate of the point as a percentage of the image width")
    y: Optional[float] = Field(None, description="Y-coordinate of the point as a percentage of the image height")
    name: Optional[str] = Field(None, description="Object name")
    validity: ValidityStatus = Field(default=ValidityStatus.UNKNOWN)

class ImageObject(BaseModel):
    id: str = Field(default_factory=uuid.uuid4)
    name: Optional[str] = None
    description: Optional[str] = None
    category: Optional[ObjectCategory] = None
    bbox: Optional[BoundingBox] = None
    mask: Optional[SegmentationMask] = None
    significance: Optional[str] = None
    validity: ObjectValidity = Field(default=ObjectValidity())
    clickpoint: Optional[ClickPoint] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

class ClickingImage(BaseModel):
    image: Image.Image
    path: Optional[str] = None
    id: str
    ui_elements: List[UIElement] = Field(default_factory=list)
    annotated_objects: List[ImageObject] = Field(default_factory=list)
    predicted_objects: List[ImageObject] = Field(default_factory=list)
    ui_elements: List[UIElement] = Field(default_factory=list)
    model_config = ConfigDict(arbitrary_types_allowed=True)


class ModuleMode(NamedTuple):
    name: str
    handler: Callable[[object], str]

class ObjectImageDict(BaseModel):
    image_id: str
    object: ImageObject

    model_config = ConfigDict(arbitrary_types_allowed=True)


@dataclass
class PipelineState:
    images: List[ClickingImage] = field(default_factory=list)

    def __hash__(self):
        # Create a hash based on the content of PipelineState
        return hash(pickle.dumps(self))

    def __eq__(self, other):
        if not isinstance(other, PipelineState):
            return False
        return self.images == other.images

    def filter_by_object_category(self, category: ObjectCategory):
        for clicking_image in self.images:
            clicking_image.predicted_objects = [
                obj for obj in clicking_image.predicted_objects
                if obj.category == category
            ]
        return self

    def get_image_by_id(self, image_id: str) -> ClickingImage:
        for image in self.images:
            if image.id == image_id:
                return image
        raise ValueError(f"Image with id {image_id} not found")

    def find_object_by_id(self, object_id: str) -> Optional[ImageObject]:
        for image in self.images:
            obj = next((obj for obj in image.predicted_objects if str(obj.id) == object_id), None)
            if obj:
                return obj
        return None

    def find_ui_element_by_id(self, object_id: str) -> Optional[UIElement]:
        for image in self.images:
            obj = next((obj for obj in image.ui_elements if str(obj.id) == object_id), None)
            if obj:
                return obj
        return None

    def filter_by_ids(self, image_ids: Optional[List[int]] = None, sample_size: Optional[int] = None):
        if image_ids is not None and sample_size is not None:
            raise ValueError("Cannot specify both image_ids and sample_size. Choose one filtering method.")
        
        if image_ids is not None:
            image_ids = [str(id) for id in image_ids]
            self.images = [img for img in self.images if img.id in image_ids]
        elif sample_size is not None:
            if sample_size > len(self.images):
                raise ValueError(f"Sample size {sample_size} is larger than the number of available images {len(self.images)}")
            self.images = random.sample(self.images, sample_size)
        
        return self
    
    def get_all_predicted_objects(self) -> dict[str, ObjectImageDict]:
        all_objects = {}
        for image in self.images:
            for obj in image.predicted_objects:
                all_objects[str(obj.id)] = ObjectImageDict(image_id=str(image.id), object=obj)
        return all_objects


from pydantic import BaseModel, ConfigDict, Field
from typing import Any, Union, Optional, List
from enum import Enum, auto
from PIL import Image
import numpy as np
from fastapi import Form, File, UploadFile, Depends
from typing import NamedTuple, List, Dict
from clicking.common.bbox import BoundingBox
from clicking.common.mask import SegmentationMask

class SegmentationReq(BaseModel):  
    image: Any
    input_boxes: list

class ModelInfo(BaseModel):
    name: str
    variants: list
    tasks: list

class GetModelsResp(BaseModel):
    models: list[ModelInfo]

class GetModelResp(BaseModel):
    name: str
    variant: str

class SetModelReq(BaseModel):
    name: str
    variant: str 
    task: TaskType

class GetModelReq(BaseModel):
    task: TaskType

class LocalizationReq(BaseModel):  
    image: str
    text_input: str
    task_prompt: str

class LocalizationResp(BaseModel):
    bboxes: list
    labels: list

class SegmentationResp(BaseModel):
    masks: list
    scores: Optional[list] = None

# class PredictionReq(BaseModel):
#     image: UploadFile = Field(..., description="Uploaded image file")
#     task: TaskType
#     input_text: Optional[str] = Field(None, description="Text input for text-based tasks")
#     input_boxes: Optional[str] = Field(None)
#     input_point: Optional[str] = Field(None)
#     input_label: Optional[str] = Field(None)
#     enable_cache: Optional[bool] = Field(True)
#     reset_cache: Optional[bool] = Field(False)

#     model_config = ConfigDict(arbitrary_types_allowed=True)

# class BatchPredictionReq(BaseModel):
#     requests: List[PredictionReq] = Field(None, description="List of prediction requests")

#     model_config = ConfigDict(arbitrary_types_allowed=True)

class PredictionReq(BaseModel):
    id: Optional[str] = None
    image: str = Field(..., description="Base64 encoded image string")
    task: TaskType
    input_text: Optional[str] = Field(None, description="Text input for text-based tasks")
    input_boxes: Optional[str] = Field(None)
    input_point: Optional[str] = Field(None)
    input_label: Optional[str] = Field(None)
    enable_cache: Optional[bool] = Field(True)
    reset_cache: Optional[bool] = Field(False)

class BatchPredictionReq(BaseModel):
    requests: List[PredictionReq] = Field(..., description="List of prediction requests")

class PredictionResp(BaseModel):
    id: Optional[str] = None
    inference_time: Optional[float] = 0.0
    prediction: Union[LocalizationResp, SegmentationResp, ClickPoint]

class BatchPredictionResp(BaseModel):
    responses: List[PredictionResp]
    inference_time: Optional[float] = 0.0

class AutoAnnotationReq(BaseModel):
    image: UploadFile = Field(..., description="Uploaded image file")
    task: TaskType
    points_per_side: Optional[int] = Field(32)
    points_per_batch: Optional[int] = Field(64)
    pred_iou_thresh: Optional[float] = Field(0.8)
    stability_score_thresh: Optional[float] = Field(0.95)
    stability_score_offset: Optional[float] = Field(1.0)
    mask_threshold: Optional[float] = Field(0.0)
    box_nms_thresh: Optional[float] = Field(0.7)
    crop_n_layers: Optional[int] = Field(0)
    crop_nms_thresh: Optional[float] = Field(0.7)
    crop_overlap_ratio: Optional[float] = Field(512 / 1500)
    crop_n_points_downscale_factor: Optional[int] = Field(1)
    # point_grids: Optional[List[np.ndarray]] = None,
    min_mask_region_area: Optional[int] = Field(0)
    output_mode: Optional[str] = Field("coco_rle")
    use_m2m: Optional[bool] = Field(False)
    multimask_output: Optional[bool] = Field(True)

    model_config = ConfigDict(arbitrary_types_allowed=True)

class AutoAnnotationResp(BaseModel):
    prediction: SegmentationResp
    inference_time: Optional[float] = 0.0

class OCRResult(BaseModel):
    label: str
    quad_box: List[float]