from clicking.localization.model import Florence2
from pydantic import BaseModel
from PIL import Image
import io
import base64
import time
from typing import Dict, List, Type
from dataclasses import dataclass, field

class LocalizationRequest(BaseModel):  
    image: str
    text_input: str
    task_prompt: str

class LocalizationResp(BaseModel):
    bboxes: list
    labels: list
    inference_time: float

@dataclass
class ModelInfo:
    name: str
    variants: List[str]
    class_: Type

class LocalizationModel:
    def __init__(self):
        self._model = None
        self._available_models = {
            'florence2': ModelInfo('florence2', ['florence-2-base','florence-2-large'], Florence2),
            # Add more models here as needed
        }
    def get_model(self):
        if self._model is None:
            raise ValueError("Model not set")
        return self._model

    def set_model(self, model_name: str, model_variant: str):
        if model_name not in self._available_models:
            raise ValueError(f"Model {model_name} not supported")
        
        model_info = self._available_models[model_name]
        if model_variant not in model_info.variants:
            raise ValueError(f"Variant {model_variant} not supported for model {model_name}")
        
        self._model = model_info.class_(model_variant)

        print(f"Localization model set to {model_name} with variant {model_variant}.")

    def get_available_models(self):
        return {name: info.variants for name, info in self._available_models.items()}

    async def get_localization(self, req: LocalizationRequest):
        base64_image = req.image
        text_input = req.text_input
        task_prompt = req.task_prompt

        # Convert base64 string back to image
        image = base64.b64decode(base64_image)
        image = Image.open(io.BytesIO(image))

        # run inference and measure execution time
        start_time = time.time()
        response = self._model.run_inference(image, task_prompt, text_input=text_input)
        end_time = time.time()
        inference_time = end_time - start_time

        response['inference_time'] = inference_time
        return response   

# localization_model = LocalizationModel()
# print('Available models:', localization_model.get_available_models())

