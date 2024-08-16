from fastapi import APIRouter, File, UploadFile, Form
from clicking.segmentation.core import PredictionReq, PredictionResp, SegmentationModel, GetModelsResp, GetModelResp, SetModelRequest
from PIL import Image
import io
import json

segmentation_router = APIRouter()
model = SegmentationModel()
# response_model=PredictionResp, 

@segmentation_router.post("/prediction",operation_id="get_segmentation_prediction")


async def semantic_segmentation(image: UploadFile = File(...),
    task_prompt: str = Form(None),
    input_boxes: str = Form(None)) -> PredictionResp:

    #Convert to a PIL image
    image_data = await image.read()
    image = Image.open(io.BytesIO(image_data))

    print("task_prompt", task_prompt)
    # Convert input_boxes from string to float list\
    input_boxes = json.loads(input_boxes) if input_boxes else []
    
    print("input_boxes", input_boxes)
    
    if task_prompt == "bbox":
        response = await model.get_segmentation_prediction(image, input_boxes)
    else:
        raise ValueError(f"Invalid task prompt: {task_prompt}")

    return response

@segmentation_router.get("/models", response_model=GetModelsResp, operation_id="get_available_models")
async def get_models():
    models = model.get_available_models()
    return models

@segmentation_router.get("/model", response_model=GetModelResp, operation_id="get_model")
async def get_model():
    model = model.get_model()
    return model

@segmentation_router.post("/model", operation_id="set_model")
async def set_model(req: SetModelRequest):
    try:
        model.set_model(req)
        return {"status": "OK"}
    except ValueError as e:
        return {"status": "error", "message": str(e)}