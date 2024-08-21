from fastapi import APIRouter, Query, File, UploadFile, Form, Path, HTTPException
from clicking.vision_model.core import SegmentationReq, SegmentationResp, LocalizationReq, LocalizationResp, VisionModel, GetModelsResp, GetModelResp, SetModelRequest, TaskType
from PIL import Image
import io
import json

vision_model_router = APIRouter()
vision_model = VisionModel()

# # Create a mapping dictionary
# TASK_TYPE_MAP = {
#     "localization": TaskType.LOCALIZATION,
#     "segmentation": TaskType.SEGMENTATION
# }

@vision_model_router.get("/models", response_model=GetModelsResp, operation_id="get_models")
async def get_models():
    models = vision_model.get_available_models()
    return models


# @vision_model_router.get("/model/{task_type}", response_model=GetModelResp, operation_id="get_model")
# async def get_model(task_type: str = Path(..., description="Type of task"), set_model_request: SetModelRequest = None):
#     # Convert task_type string to TaskType enum
#     task_type = TASK_TYPE_MAP.get(task_type.lower())
    
#     if task_type is None:
#         raise HTTPException(status_code=400, detail=f"Invalid task type: {task_type}")
    
#     # Get model info using the TaskType enum
#     model_info = vision_model.get_model(task_type)
#     return model_info


@vision_model_router.post("/model", operation_id="set_model")
async def set_model(req: SetModelRequest = None):

    # Get model info using the TaskType enum
    result = vision_model.set_model(req)
    print(req)
    return {"message": "Model set successfully", "status_code": 200}

@vision_model_router.get("/prediction/localization", response_model=LocalizationResp, operation_id="get_localization_prediction")
async def get_prediction(
    image: str,
    text_input: str,
    task_prompt: str,
    task_type: TaskType = Query(..., description="Type of task to perform")
):
    req = LocalizationReq(image=image, text_input=text_input, task_prompt=task_prompt)
    result, inference_time = await vision_model.get_localization(req.image, req.text_input, req.task_prompt, task_type)

    response = LocalizationResp(
        bboxes=result['bboxes'],
        labels=result['labels'],
        inference_time=inference_time
    )

    return response

@vision_model_router.post("/prediction/segmentation",operation_id="get_segmentation_prediction", response_model=SegmentationResp)
async def semantic_segmentation(image: UploadFile = File(...),
    task_prompt: str = Form(None),
    input_boxes: str = Form(None)):

    #Convert to a PIL image
    image_data = await image.read()
    image = Image.open(io.BytesIO(image_data))

    print("task_prompt", task_prompt)

    input_boxes = json.loads(input_boxes) if input_boxes else []
    response = await vision_model.get_segmentation_prediction(image, input_boxes)

    return response