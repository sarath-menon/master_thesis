from fastapi import APIRouter, Query, File, UploadFile, Form, Path, HTTPException
from clicking.vision_model.core import *
from PIL import Image
import io
import json
from fastapi import Depends
from typing import Dict, Tuple, Optional
from ..cache_macro import cache_prediction
import hashlib

vision_model_router = APIRouter()
vision_model = VisionModel()

# Cache to store recent predictions
prediction_cache: Dict[str, Tuple[PredictionResp, float]] = {}
CACHE_EXPIRATION_TIME = 300  # 5 minutes in seconds

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


@vision_model_router.post("/prediction", operation_id="get_prediction", response_model=PredictionResp)
@cache_prediction
async def prediction(
    image: UploadFile = File(...),
    task: TaskType = Form(None),
    input_boxes: str = Form(None),
    input_point: str = Form(None),
    input_label: str = Form(None),
    input_text: str = Form(None),
    enable_cache: Optional[bool] = Form(True),
    reset_cache: Optional[bool] = Form(False),
):
    if task is None:
        raise HTTPException(status_code=400, detail="Task is required")

    return await process_prediction(image, task, input_boxes, input_point, input_label, input_text)

async def process_prediction(image, task, input_boxes, input_point, input_label, input_text):
    image_data = await image.read()
    image = Image.open(io.BytesIO(image_data))

    if input_boxes is not None:
        input_boxes = json.loads(input_boxes)

    req = PredictionReq(image=image, task=task, input_point=input_point, input_label=input_label, input_box=input_boxes, input_text=input_text)
    
    return await vision_model.get_prediction(req)

def generate_cache_key(*args) -> str:
    key = hashlib.md5()
    for arg in args:
        if isinstance(arg, UploadFile):
            key.update(arg.filename.encode())
        elif arg is not None:
            key.update(str(arg).encode())
    return key.hexdigest()

@vision_model_router.post("/auto_annotation", operation_id="get_auto_annotation")
async def auto_annotation(req: AutoAnnotationReq = Depends()) -> AutoAnnotationResp:
    if req.task is None:
        raise HTTPException(status_code=400, detail="Task is required")

    response = await vision_model.get_auto_annotation(req)
    return response