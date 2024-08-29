from clicking.vision_model.florence2 import Florence2
from clicking.vision_model.sam2 import SAM2
# from clicking.vision_model.evf_sam2 import EVF_SAM
from pydantic import BaseModel
from PIL import Image
import io
import base64
import time
from typing import Dict, List, Type, Any
from dataclasses import dataclass, field
from fastapi import HTTPException
from clicking.vision_model.bbox import BoundingBox, BBoxMode
import numpy as np
from enum import Enum, auto
from clicking.vision_model.types import *

class VisionModel:
    def __init__(self):
        self._available_models = {
            'florence2': Florence2,
            'sam2': SAM2,
            # 'evf_sam2': EVF_SAM
        }
        # to store task-model mappings
        self._task_models = {}  

    def tasks(self) -> list[str]:
        return [task.value for task in TaskType]

    def _get_model_for_task(self, task: TaskType):
        if task not in self._task_models:
            return None
        return self._task_models[task]

    def get_model(self, task: TaskType) ->GetModelResp:
        # check if task-model mapping exists
        model = self._get_model_for_task(task)

        if model is None:
            raise HTTPException(status_code=404, detail=f"{task.name.capitalize()} model not set")
        
        response = GetModelResp(name=model.name, variant=model.variant)
        return response


    def set_model(self, req: SetModelReq):
        if req.name not in self._available_models:
            raise HTTPException(status_code=400, detail=f"Model {req.name} not supported")
        
        model_class_obj = self._available_models[req.name]
        
        if req.variant not in model_class_obj.variants():
            raise HTTPException(status_code=400, detail=f"Variant {req.variant} not supported for model {req.name}")

        task_type = TaskType(req.task)

        # Check if model has the task
        if task_type not in model_class_obj.tasks():
            raise HTTPException(status_code=400, detail=f"Model {req.name} does not support task {req.task}")

        model_handle = model_class_obj(variant=req.variant)
        
        # Save the model for the task
        self._task_models[task_type] = model_handle
        
        message = f"{req.task.capitalize()} model set to {req.name} with variant {req.variant}."
        return {"message": message, "status_code": 200}

    def get_available_models(self):
        models = []
        for name, handle in self._available_models.items():
            model = ModelInfo(name=name, variants=handle.variants(), tasks=handle.tasks())
            models.append(model)
        return GetModelsResp(models=models)


    async def get_prediction(self, req: PredictionReq) -> PredictionResp:
        model_handle = self._get_model_for_task(req.task)
        if model_handle is None:
            raise HTTPException(status_code=404, detail=f"{req.task.name.capitalize()} model not set")

        start_time = time.time()
        response = model_handle.predict(req)
        response.inference_time = time.time() - start_time

        return response


    async def get_auto_annotation(self, req: AutoAnnotationReq) -> PredictionResp:
        model_handle = self._get_model_for_task(req.task)
        if model_handle is None:
            raise HTTPException(status_code=404, detail=f"{req.task.name.capitalize()} model not set")

        start_time = time.time()
        response = await model_handle.auto_annotate(req)
        response.inference_time = time.time() - start_time

        return response
