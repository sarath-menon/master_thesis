from clicking.vision_model.florence2 import Florence2
from clicking.vision_model.sam2 import SAM2
from pydantic import BaseModel
from PIL import Image
import io
import base64
import time
from typing import Dict, List, Type, Any
from dataclasses import dataclass, field
from fastapi import HTTPException
from clicking.visualization.bbox import BoundingBox, BBoxMode
from pycocotools import mask as mask_utils
import numpy as np
from enum import Enum, auto
from clicking.vision_model.types import *

class VisionModel:
    def __init__(self):
        self._available_models = {
            'florence2': Florence2,
            'sam2': SAM2,
        }
        # to store task-model mappings
        self._task_models = {}  

    def tasks(self) -> list[str]:
        return [task.value for task in TaskType]

    def get_model(self, task: TaskType) ->GetModelResp:
        # check if task-model mapping exists
        if task not in self._task_models:
            raise HTTPException(status_code=404, detail=f"{task.name.capitalize()} model not set")

        model = self._task_models[task]
        response = GetModelResp(name=model.name, variant=model.variant)
        return response


    def set_model(self, req: SetModelReq):
        if req.name not in self._available_models:
            raise HTTPException(status_code=400, detail=f"Model {req.name} not supported")
        
        model_handle = self._available_models[req.name]
        
        if req.variant not in model_handle.variants():
            raise HTTPException(status_code=400, detail=f"Variant {req.variant} not supported for model {req.name}")

        task_type = TaskType(req.task)

        # Check if model has the task
        if task_type not in model_handle.tasks():
            raise HTTPException(status_code=400, detail=f"Model {req.name} does not support task {req.task}")

        model = model_handle(req.variant)
        
        # Save the model for the task
        self._task_models[task_type] = model
        
        message = f"{req.task.capitalize()} model set to {req.name} with variant {req.variant}."
        return {"message": message, "status_code": 200}

    def get_available_models(self):
        models = []
        for name, handle in self._available_models.items():
            model = ModelInfo(name=name, variants=handle.variants(), tasks=handle.tasks())
            models.append(model)
        return GetModelsResp(models=models)

    # def _get_model_for_task(self, task: TaskType):

    #     else:
    #         raise HTTPException(status_code=400, detail=f"Invalid task: {task}")

    def coco_encode_rle(self, mask: np.ndarray) -> Dict[str, Any]:
        binary_mask = mask.astype(bool)
        rle = mask_utils.encode(np.asfortranarray(binary_mask))
        rle['counts'] = rle['counts'].decode('utf-8')
        return rle

    async def get_prediction(self, task: TaskType, image_data, text_input=None, task_prompt=None, input_boxes=None):
        model = self._get_model_for_task(task)
        if model is None:
            raise HTTPException(status_code=404, detail=f"{task.name.capitalize()} model not set")

        if task == TaskType.LOCALIZATION:
            image = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image))
            start_time = time.time()
            result = model.run_inference(image, task_prompt, text_input=text_input)
            inference_time = time.time() - start_time
            return (result, inference_time)

        elif task == TaskType.SEGMENTATION:
            start_time = time.time()
            masks, scores = model.predict_with_batched_bbox(image_data, input_boxes)
            inference_time = time.time() - start_time
            scores = scores.tolist()
            masks = [self.coco_encode_rle(mask) for mask in masks]
            return SegmentationResp(masks=masks, scores=scores, inference_time=inference_time)
        else:
            raise HTTPException(status_code=400, detail=f"Invalid task: {task}")

