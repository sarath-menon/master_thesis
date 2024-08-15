from fastapi import APIRouter
from clicking.localization.core import PredictionReq, PredictionResp, LocalizationModel, GetModelsResp, GetModelResp, SetModelRequest

localization_router = APIRouter()
localization_model = LocalizationModel()

@localization_router.get("/prediction", response_model=PredictionResp, operation_id="get_localization_prediction")
async def get_prediction(req: PredictionReq):
    return await localization_model.get_localization(req)

@localization_router.get("/models", response_model=GetModelsResp, operation_id="get_available_localization_models")
async def get_models():
    models = localization_model.get_available_models()
    return models

@localization_router.get("/model", response_model=GetModelResp, operation_id="get_localization_model")
async def get_model():
    model = localization_model.get_model()
    return model

@localization_router.post("/model", operation_id="set_localization_model")
async def set_model(req: SetModelRequest):
    try:
        localization_model.set_model(req)
        return {"status": "OK"}
    except ValueError as e:
        return {"status": "error", "message": str(e)}