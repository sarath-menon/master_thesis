from pydantic import BaseModel
from PIL import Image
import io
import base64
import time
from typing import Dict, List, Type, Any
from dataclasses import dataclass, field
from fastapi import HTTPException
from clicking.common.bbox import BoundingBox, BBoxMode
import numpy as np
from enum import Enum, auto
from clicking.common.data_structures import *

import asyncio
import logging

class VisionModel:
    def __init__(self):
        self._available_models = ['florence2', 'sam2', 'evf_sam2', 'molmo']
        
        # to store task-model mappings
        self._task_models = {}  
        self._model_locks = {}
        self.logger = logging.getLogger("uvicorn")

    def tasks(self) -> list[str]:
        return [task.value for task in TaskType]

    def _get_model_for_task(self, task: TaskType):
        if task not in self._task_models:
            return None
        return self._task_models[task]

    def get_model(self, task: TaskType) ->GetModelResp:
        # check if task-model mapping exists
        model = self._get_model_for_task(task)

        response = GetModelResp(name=model.name, variant=model.variant)
        return response


    def set_model(self, req: SetModelReq):
        task_type = TaskType(req.task)

        self.logger.debug(f"self._task_models: {self._task_models}")
        
        # Check if the model is already set for the task
        if task_type in self._task_models and self._task_models[task_type].name == req.name:
            message = f"Model {req.name} is already set for task {req.task}"
            return {"message": message, "status_code": 200}

        if req.name == 'evf_sam2':
            from clicking.vision_model.evf_sam2 import EVF_SAM
            model_class_obj = EVF_SAM
        elif req.name == 'florence2':
            from clicking.vision_model.florence2 import Florence2
            model_class_obj = Florence2
        elif req.name == 'sam2':
            from clicking.vision_model.sam2 import SAM2
            model_class_obj = SAM2
        elif req.name == 'molmo':
            from clicking.vision_model.molmo import Molmo
            model_class_obj = Molmo
        else:
            raise HTTPException(status_code=400, detail=f"Model {req.name} not supported")
        
        if req.variant not in model_class_obj.variants():
            print(f"Variant {req.variant} not supported for model {req.name}")
            raise HTTPException(status_code=400, detail=f"Variant {req.variant} not supported for model {req.name}")

        task_type = TaskType(req.task)
        if task_type not in model_class_obj.tasks():
            print(f"Model {req.name} does not support task {req.task}")
            raise HTTPException(status_code=400, detail=f"Model {req.name} does not support task {req.task}")

        model_handle = model_class_obj(variant=req.variant)
        
        try:
            self._task_models[task_type] = model_handle
            message = f"{req.task.capitalize()} model set to {req.name} with variant {req.variant}."
            return {"message": message, "status_code": 200}
        except Exception as e:
            error_message = f"Failed to set model for task {req.task}: {str(e)}"
            return {"message": error_message, "status_code": 500}

    def get_available_models(self):
        models = []
        for name, handle in self._available_models.items():
            model = ModelInfo(name=name, variants=handle.variants(), tasks=handle.tasks())
            models.append(model)
        return GetModelsResp(models=models)  # Change this line

    async def get_prediction(self, req: PredictionReq):
        model_handle = self._get_model_for_task(req.task)

        if model_handle is None:
            print(f"Model not set: {model_handle}")
            raise HTTPException(status_code=404, detail=f"{req.task.name.capitalize()} model not set")
        
        if req.task not in self._model_locks:
            self._model_locks[req.task] = asyncio.Semaphore(1)

        try:
            async with self._model_locks[req.task]:
                start_time = time.time()
                response = await model_handle.predict(req)
                response.inference_time = time.time() - start_time
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")

        return response


    async def get_auto_annotation(self, req: AutoAnnotationReq) -> PredictionResp:
        model_handle = self._get_model_for_task(req.task)
        if model_handle is None:
            raise HTTPException(status_code=404, detail=f"{req.task.name.capitalize()} model not set")

        start_time = time.time()
        response = await model_handle.auto_annotate(req)
        response.inference_time = time.time() - start_time

        return response