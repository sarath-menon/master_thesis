from fastapi import APIRouter, Query, File, UploadFile, Form, Path, HTTPException, Depends
from fastapi.responses import StreamingResponse
from clicking.common.data_structures import *
from clicking.api.exceptions import ServiceNotAvailableException, ModelNotSetException
from clicking.vision_model.core import VisionModel
from PIL import Image
from typing import Dict, Tuple, Optional, AsyncIterable
from ..cache_macro import cache_prediction
import time
import asyncio
import json

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

    response = await vision_model.get_prediction(req)
    return response

@vision_model_router.post("/auto_annotation", operation_id="get_auto_annotation")
async def auto_annotation(req: AutoAnnotationReq = Depends()) -> AutoAnnotationResp:
    if req.task is None:
        raise HTTPException(status_code=400, detail="Task is required")

    response = await vision_model.get_auto_annotation(req)
    return response

@vision_model_router.post("/batch_prediction", operation_id="get_batch_prediction", response_model=BatchPredictionResp)
@cache_prediction
async def batch_prediction(batch_requests: BatchPredictionReq = Depends()) -> BatchPredictionResp:
    if not batch_requests:
        raise HTTPException(status_code=400, detail="Batch request is empty")

    start_time = time.time()

    # Print the body size of the request in megabytes
    request_size = len(batch_requests.json()) / (1024 * 1024)
    print(f"Request body size: {request_size} MB")

    responses = []
    for req in batch_requests.requests:
        print(f"Processing request {req.id}")
        if req.task is None:
            raise HTTPException(status_code=400, detail="Task is required for all requests in the batch")

        response = await vision_model.get_prediction(req)
        response.id = req.id
        responses.append(response)

    end_time = time.time()
    total_time = end_time - start_time

    return BatchPredictionResp(responses=responses, inference_time=total_time)

    
@vision_model_router.post("/stream_batch_prediction", operation_id="stream_batch_prediction")
async def stream_batch_prediction(batch_requests: BatchPredictionReq = Depends()) -> BatchPredictionResp:
    if not batch_requests:
        raise HTTPException(status_code=400, detail="Batch request is empty")

    start_time = time.time()
    responses = []

    async def process_requests():
        nonlocal responses
        for req in batch_requests.requests:
            if req.task is None:
                raise HTTPException(status_code=400, detail="Task is required for all requests in the batch")

            print(f"Processing request {req.id}")
            response = await vision_model.get_prediction(req)
            response.id = req.id
            responses.append(response)

            # Add a small delay to simulate streaming
            await asyncio.sleep(0.01)

    # Process requests asynchronously
    await process_requests()

    end_time = time.time()
    total_time = end_time - start_time

    return BatchPredictionResp(responses=responses, inference_time=total_time)