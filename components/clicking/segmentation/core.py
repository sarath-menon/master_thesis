from clicking.segmentation.sam2 import SAM2
from pydantic import BaseModel
from PIL import Image
import io
import base64
import time
from typing import Dict, List, Type, Any
from dataclasses import dataclass, field
from pycocotools import mask as mask_utils
import numpy as np
from fastapi import HTTPException


class PredictionReq(BaseModel):  
    image: Any
    input_boxes: list

class SegmentationPredResp(BaseModel):
    masks: list
    scores: list
    inference_time: float

class ModelInfo(BaseModel):
    name: str
    variants: list
    tasks: list

class GetModelsResp(BaseModel):
    models: list[ModelInfo]

class GetModelResp(BaseModel):
    name: str
    variant: str

class SetModelReq(BaseModel):
    name: str
    variant: str 

class SegmentationModel:
    def __init__(self):
        self._model = None
        self._available_models = {
            'sam2': SAM2,
        }

    def get_model(self) -> GetModelResp:
        if self._model is None:
            raise HTTPException(status_code=404, detail="Model not set")
        return GetModelResp(name=self._model.name, variant=self._model.variant)

    def set_model(self, req: SetModelReq):
        if req.name not in self._available_models:
            raise HTTPException(status_code=400, detail=f"Model {req.name} not supported")
        
        model_handle = self._available_models[req.name]
        if req.variant not in model_handle.variants():
            raise HTTPException(status_code=400, detail=f"Variant {req.variant} not supported for model {req.name}")
        
        self._model = model_handle(req.variant)

        message = f"Segmentation model set to {req.name} with variant {req.variant}."
        print(message)
        return {"message": message, "status_code": 200}

    def get_available_models(self):
        models = []
        for name in self._available_models.keys():
            handle = self._available_models[name]
            model = ModelInfo(name=name, variants=handle.variants(), tasks=handle.tasks())
            models.append(model)
        return GetModelsResp(models=models) 

    def coco_encode_rle(self, mask: np.ndarray) -> Dict[str, Any]:
        # Ensure the mask is boolean
        binary_mask = mask.astype(bool)
        
        # Encode the mask
        rle = mask_utils.encode(np.asfortranarray(binary_mask))
        rle['counts'] = rle['counts'].decode('utf-8')
        return rle

    async def get_segmentation_prediction(self, image, input_boxes) -> SegmentationPredResp:
        if self._model is None:
            raise HTTPException(status_code=404, detail="Model not set")

        # run inference and measure execution time
        start_time = time.time()
        # result = self._model.run_inference(image, task_prompt, text_input=text_input)
        masks, scores = self._model.predict_with_batched_bbox(image, input_boxes)
        end_time = time.time()
        inference_time = end_time - start_time

        # convert masks and scores from numpy arrays to lists
        scores = scores.tolist()

        #convert masks to coco rle
        masks = [self.coco_encode_rle(mask) for mask in masks]

        return SegmentationPredResp(masks=masks, scores=scores, inference_time=inference_time)

# localization_model = LocalizationModel()
# print('Available models:', localization_model.get_available_models())

