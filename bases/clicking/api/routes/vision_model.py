from fastapi import APIRouter, Query, File, UploadFile, Form, Path, HTTPException, Depends
from clicking.common.data_structures import *
from clicking.api.exceptions import ServiceNotAvailableException, ModelNotSetException
from clicking.vision_model.core import VisionModel
from PIL import Image
import io
import json
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
async def prediction(req: PredictionReq = Depends()) -> PredictionResp:
    if req.task is None:
        raise HTTPException(status_code=400, detail="Task is required")

    try:
        response = await vision_model.get_prediction(req)
        print(response)
        return response
    except ModelNotSetException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ServiceNotAvailableException as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@vision_model_router.post("/auto_annotation", operation_id="get_auto_annotation")
async def auto_annotation(req: AutoAnnotationReq = Depends()) -> AutoAnnotationResp:
    if req.task is None:
        raise HTTPException(status_code=400, detail="Task is required")

    response = await vision_model.get_auto_annotation(req)
    return response