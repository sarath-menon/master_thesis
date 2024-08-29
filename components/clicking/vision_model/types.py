from pydantic import BaseModel, ConfigDict, Field
from typing import Any, Union, Optional, List
from enum import Enum, auto
from PIL import Image
import numpy as np
from fastapi import Form, File, UploadFile, Depends
from typing import NamedTuple, List, Dict
from clicking.prompt_refinement.core import ProcessedSample
from clicking.vision_model.bbox import BoundingBox

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

class PredictionReq(BaseModel):
    image: Image.Image
    task: TaskType
    input_point: Optional[list] = None
    input_label: Optional[list] = None
    input_box: Optional[list] = None
    input_text: Optional[str] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

class PredictionResp(BaseModel):
    inference_time: Optional[float] = 0.0
    prediction: Union[LocalizationResp, SegmentationResp]

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

class LocalizationResults(NamedTuple):
    processed_samples: List[ProcessedSample]
    predictions: Dict[str, List[BoundingBox]] 