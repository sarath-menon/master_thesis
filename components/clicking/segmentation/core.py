from clicking.segmentation.sam2 import SAM2
from pydantic import BaseModel
from PIL import Image
import io
import base64
import time
from typing import Dict, List, Type
from dataclasses import dataclass, field

class PredictionReq(BaseModel):  
    image: str
    text_input: str
    task_prompt: str

class PredictionResp(BaseModel):
    bboxes: list
    labels: list
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

class SetModelRequest(BaseModel):
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
            raise ValueError("Model not set")
        return GetModelResp(name=self._model.name, variant=self._model.variant)

    def set_model(self, req: SetModelRequest):
        if req.name not in self._available_models:
            raise ValueError(f"Model {req.name} not supported")
        
        model_handle = self._available_models[req.name]
        if req.variant not in model_handle.variants():
            raise ValueError(f"Variant {req.variant} not supported for model {req.name}")
        
        self._model = model_handle(req.variant)

        message = f"Segmentation model set to {req.name} with variant {req.variant}."
        print(message)

    def get_available_models(self):
        models = []
        for name in self._available_models.keys():
            handle = self._available_models[name]
            model = ModelInfo(name=name, variants=handle.variants(), tasks=handle.tasks())
            models.append(model)
        return GetModelsResp(models=models) 

    async def get_segmentation_prediction(self, base64_image: str, text_input: str, task_prompt: str) :
        # Convert base64 string back to image
        image = base64.b64decode(base64_image)
        image = Image.open(io.BytesIO(image))

        # run inference and measure execution time
        start_time = time.time()
        result = self._model.run_inference(image, task_prompt, text_input=text_input)
        end_time = time.time()
        inference_time = end_time - start_time

        return (result, inference_time)

# localization_model = LocalizationModel()
# print('Available models:', localization_model.get_available_models())

