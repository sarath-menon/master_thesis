from transformers import AutoProcessor, AutoModelForCausalLM, GenerationConfig
from PIL import Image
import os
import numpy as np
from transformers.dynamic_module_utils import get_imports
import torch
from dataclasses import dataclass
from clicking.common.data_structures import *
import io
import base64
from .utils import base64_to_pil
import re
from vllm import LLM, SamplingParams

class Molmo():
    variant_to_id = {
        'molmo-7B': "allenai/Molmo-7B-D-0924",
        'molmo-72B': "allenai/Molmo-72B-0924"
        }

    task_prompts = {
        TaskType.CLICKPOINT_WITH_TEXT: ""
        }

    def __init__(self, variant='molmo-7B'):
        self.name = 'molmo'
        self.variant = variant
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # self.processor, self.model = self.load_model(self.variant)
        self.model = self.load_model(self.variant)

    @staticmethod
    def variants():
        return list(Molmo.variant_to_id.keys())
    
    @staticmethod
    def tasks():
        return list(Molmo.task_prompts.keys())
        
    def load_model(self, model_id):
        # # huggingface model loading
        # processor = AutoProcessor.from_pretrained(
        #     self.variant_to_id[model_id],
        #     trust_remote_code=True,
        #     torch_dtype='auto',
        #     device_map='auto'
        #     )

        # # load the model
        # model = AutoModelForCausalLM.from_pretrained(
        #     self.variant_to_id[model_id],
        #     trust_remote_code=True,
        #     torch_dtype='auto',
        #     device_map='auto'
        # )
        # return processor, model

        # vllm model loading
        model = LLM(self.variant_to_id[model_id], trust_remote_code=True)
        return model

    async def predict(self, req: PredictionReq):
        if req.task not in self.task_prompts:
            raise ValueError(f"Invalid task type: {req.task}")
        elif req.image is None:
            raise ValueError("Image is required for any vision task")

        # convert base64 to PIL image
        image_pil = base64_to_pil(req.image)


        response = self.run_inference(image_pil, req.task, req.input_text)
        return PredictionResp(prediction=response)

    def text_to_image_point(self, text: str):
        pattern = r'<point x="(\d+(?:\.\d+)?)" y="(\d+(?:\.\d+)?)" alt="([^"]+)">'
        match = re.search(pattern, text)
        
        if not match:
            print("Invalid text format. Expected <point x=\"...\" y=\"...\" alt=\"...\">")
            return ClickPoint(validity=ClickPointValidity(status=ValidityStatus.INVALID, reason=text))
        
        x = float(match.group(1))
        y = float(match.group(2))
        alt = match.group(3)
        
        return ClickPoint(x=x, y=y, name=alt, validity=ClickPointValidity(status=ValidityStatus.VALID, reason=text))


    def run_inference(self, image, task, text_input=None):

        # # huggingface inference
        # inputs = self.processor.process(
        #     images=[image],
        #     text=text_input
        # )

        # # move inputs to the correct device and make a batch of size 1
        # inputs = {k: v.to(self.model.device).unsqueeze(0) for k, v in inputs.items()}

        # with torch.autocast(device_type="cuda", enabled=True, dtype=torch.float32):
        #     output = self.model.generate_from_batch(
        #         inputs,
        #         GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
        #         tokenizer=self.processor.tokenizer
        #     )

        # # only get generated tokens; decode them to text
        # generated_tokens = output[0,inputs['input_ids'].size(1):]
        # generated_text = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # vllm inference
        sampling_params = SamplingParams(temperature=1.0, max_tokens=200)
        outputs = self.model.generate({
            "prompt": text_input,
            "multi_modal_data": {"image": image},
        }, sampling_params)
        generated_text = outputs[0].outputs[0].text

        print("Molmo output: ", generated_text)

        if task == TaskType.CLICKPOINT_WITH_TEXT:
            return self.text_to_image_point(generated_text)
        else:
            raise ValueError(f"Invalid task type: {task}")

