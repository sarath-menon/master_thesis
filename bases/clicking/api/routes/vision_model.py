from fastapi import APIRouter, Query, File, UploadFile, Form, Path, HTTPException
from clicking.vision_model.core import *
from PIL import Image
import io
import json
import time

vision_model_router = APIRouter()
vision_model = VisionModel()


@vision_model_router.get("/models", response_model=GetModelsResp, operation_id="get_models")
async def get_models():
    models = vision_model.get_available_models()
    return models

@vision_model_router.get("/model", response_model=GetModelResp, operation_id="get_model")
async def get_model(req: GetModelReq = None):
    
    # Get model info using the TaskType enum
    response  = vision_model.get_model(req.task)
    return response


@vision_model_router.post("/model", operation_id="set_model")
async def set_model(req: SetModelReq = None):
    # Get model info using the TaskType enum
    result = vision_model.set_model(req)
    print(req)
    return {"message": "Model set successfully", "status_code": 200}


@vision_model_router.post("/prediction",operation_id="get_prediction", response_model=PredictionResp)
async def prediction(image: UploadFile = File(...),
    task: TaskType = Form(None),
    input_boxes: str = Form(None),
    input_point: str = Form(None),
    input_label: str = Form(None),
    input_text: str = Form(None),
    ):

    if task is None:
        raise HTTPException(status_code=400, detail="Task is required")

    #Convert to a PIL image
    image_data = await image.read()
    image = Image.open(io.BytesIO(image_data))

    # convert input_boxes to a list of lists
    input_boxes = json.loads(input_boxes) if input_boxes else []

    req = PredictionReq(image=image, task=task, input_point=input_point, input_label=input_label, input_box=input_boxes, input_text=input_text)
    print(req)

    
    response = await vision_model.get_prediction(req)
    return response


@vision_model_router.post("/auto_annotation",operation_id="get_auto_annotation", response_model=AutoAnnotationResp)
async def auto_annotation(image: UploadFile = File(...),
    task: TaskType = Form(None),
    min_mask_region_area: Optional[int] = Form(None),
    pred_iou_thresh: Optional[float] = Form(None),
    output_mode: Optional[str] = Form(None),
    points_per_side: Optional[int] = Form(None),
    points_per_batch: Optional[int] = Form(None),
    stability_score_thresh: Optional[float] = Form(None),
    stability_score_offset: Optional[float] = Form(None),
    mask_threshold: Optional[float] = Form(None),
    box_nms_thresh: Optional[float] = Form(None),
    crop_n_layers: Optional[int] = Form(None),
    crop_nms_thresh: Optional[float] = Form(None),
    crop_overlap_ratio: Optional[float] = Form(None),
    crop_n_points_downscale_factor: Optional[int] = Form(None),
    # point_grids: Optional[List[np.ndarray]] = None,
    use_m2m: Optional[bool] = Form(None),
    multimask_output: Optional[bool] = Form(None),
    ):

    if task is None:
        raise HTTPException(status_code=400, detail="Task is required")

    #Convert to a PIL image
    image_data = await image.read()
    image = Image.open(io.BytesIO(image_data))

    req = AutoAnnotationReq(image=image, task=task, min_mask_region_area=min_mask_region_area, pred_iou_thresh=pred_iou_thresh, output_mode=output_mode)
    print(req)

    response = await vision_model.get_auto_annotation(req)
    return response