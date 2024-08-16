from fastapi import APIRouter
from clicking.segmentation.core import PredictionReq, PredictionResp, SegmentationModel, GetModelsResp, GetModelResp, SetModelRequest

segmentation_router = APIRouter()
segmentation_model = SegmentationModel()

@segmentation_router.get("/prediction", response_model=PredictionResp, operation_id="get_segmentation_prediction")

async def get_prediction(req: PredictionReq) -> PredictionResp:
    result, inference_time = await segmentation_model.get_segmentation(req.image, req.text_input, req.task_prompt)

    response = PredictionResp(
        bboxes=result['bboxes'],
        labels=result['labels'],
        inference_time=inference_time
    )

    return response

@segmentation_router.get("/models", response_model=GetModelsResp, operation_id="get_available_segmentation_models")
async def get_models():
    models = segmentation_model.get_available_models()
    return models

@segmentation_router.get("/model", response_model=GetModelResp, operation_id="get_segmentation_model")
async def get_model():
    model = segmentation_model.get_model()
    return model

@segmentation_router.post("/model", operation_id="set_segmentation_model")
async def set_model(req: SetModelRequest):
    try:
        segmentation_model.set_model(req)
        return {"status": "OK"}
    except ValueError as e:
        return {"status": "error", "message": str(e)}