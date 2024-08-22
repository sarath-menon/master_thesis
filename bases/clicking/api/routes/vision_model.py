from fastapi import APIRouter, Query, File, UploadFile, Form, Path, HTTPException
from clicking.vision_model.core import SegmentationReq, SegmentationResp, LocalizationReq, LocalizationResp, VisionModel, GetModelsResp, GetModelResp, SetModelReq, TaskType, GetModelReq, PredictionReq, PredictionResp
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
    input_text: str = Form(None)
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