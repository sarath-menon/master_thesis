from fastapi import APIRouter
from clicking.localization.core import PredictionReq, PredictionResp, LocalizationModel, ModelsResp, ModelInfo

localization_router = APIRouter()
localization_model = LocalizationModel()

@localization_router.get("/localization", response_model=PredictionResp)
async def localization(req: PredictionReq):
    return await localization_model.get_localization(req)


@localization_router.get("/models", response_model=ModelsResp)
async def get_localization_model():
    models = localization_model.get_available_models()
    return models

@localization_router.get("/model", response_model=ModelInfo)
async def get_localization_model():
    model = localization_model.get_model()
    return {"model_name": model.name,
     "model_variant": model.variant,
     "model_tasks": model.tasks()}

@localization_router.post("/model")
async def set_localization_model(req: dict):
    model_name = req.get('model_name')
    model_variant = req.get('model_variant')
    try:
        localization_model.set_model(model_name, model_variant)
        return {"status": "OK"}, 200
    except ValueError as e:
        return {"status": "error", "message": str(e)}, 400