from fastapi import APIRouter
from clicking.localization.core import PredictionReq, PredictionResp, LocalizationModel, ModelsResp, ModelInfo

localization_router = APIRouter()
localization_model = LocalizationModel()

@localization_router.get("/prediction", response_model=PredictionResp, operation_id="get_localization_prediction")
async def get_prediction(req: PredictionReq):
    return await localization_model.get_localization(req)

@localization_router.get("/models", response_model=ModelsResp, operation_id="get_available_localization_models")
async def get_models():
    models = localization_model.get_available_models()
    return models

@localization_router.get("/model", response_model=ModelInfo, operation_id="get_localization_model")
async def get_model():
    model = localization_model.get_model()
    return {"model_name": model.name,
     "model_variant": model.variant,
     "model_tasks": model.tasks()}

@localization_router.post("/model", operation_id="set_localization_model")
async def set_model(req: dict):
    model_name = req.get('model_name')
    model_variant = req.get('model_variant')
    try:
        localization_model.set_model(model_name, model_variant)
        return {"status": "OK"}, 200
    except ValueError as e:
        return {"status": "error", "message": str(e)}, 400