#%%

from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import os
import numpy as np
from transformers.dynamic_module_utils import get_imports
import torch
from dataclasses import dataclass
from clicking.common.data_structures import *
import io
#%%
class Florence2():
    variant_to_id = {
        'florence-2-base': "microsoft/Florence-2-base",
        'florence-2-large': "microsoft/Florence-2-large"
        }

    task_prompts_map = {TaskType.LOCALIZATION_WITH_TEXT_GROUNDED: '<CAPTION_TO_PHRASE_GROUNDING>', TaskType.LOCALIZATION_WITH_TEXT_OPEN_VOCAB: '<OPEN_VOCABULARY_DETECTION>', TaskType.CAPTIONING: '<MORE_DETAILED_CAPTION>'}

    def __init__(self, variant='florence-2-base'):
        self.name = 'florence2'
        self.variant = variant
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

         # select the device for computation
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        # elif torch.backends.mps.is_available():
        #     self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        print(f"Using {self.device} for {self.name}")

        self.model, self.processor = self.load_model_gpu(self.variant_to_id[self.variant])


    @staticmethod
    def variants():
        return list(Florence2.variant_to_id.keys())
    
    @staticmethod
    def tasks():
        return list(Florence2.task_prompts_map.keys())
        
    def load_model_gpu(self, model_id):
        if self.device == 'cuda':
            model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype='auto').eval().to(self.device)
            processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            return model, processor

        else:
            from transformers.dynamic_module_utils import get_imports
            from unittest.mock import patch

            def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
                if not str(filename).endswith("/modeling_florence2.py"):
                    return get_imports(filename)
                imports = get_imports(filename)
                imports.remove("flash_attn")
                return imports

            with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
                model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to(self.device).eval()
                processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            return model, processor

    async def predict(self, req: PredictionReq) -> PredictionResp:
        if req.task not in self.task_prompts_map:
            raise ValueError(f"Invalid task type: {req.task}")
        elif req.image is None:
            raise ValueError("Image is required for any vision task")
        elif req.input_text is None:
            raise ValueError("Text input is required for florence2 vision tasks")

        #Convert to a PIL imagex
        image_pil = await req.image.read()
        image_pil = Image.open(io.BytesIO(image_pil))

        response = self.run_inference(image_pil, req.task, req.input_text)
        return PredictionResp(prediction=response)

    def run_inference(self, image, task, text_input=None) -> LocalizationResp:

        task_prompt = self.task_prompts_map[task]

        if text_input is None:
            prompt = task_prompt
        else:
            prompt = task_prompt + text_input
        # inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device, torch.float16)
        generated_ids = self.model.generate(
        input_ids=inputs["input_ids"].to(self.device),
        pixel_values=inputs["pixel_values"].to(self.device),
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
        )
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

        result = self.processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(image.width, image.height)
        )

        bboxes = []
        labels = []

        if task == TaskType.LOCALIZATION_WITH_TEXT_OPEN_VOCAB:
            bboxes = result[task_prompt]["bboxes"]
            labels = result[task_prompt]["bboxes_labels"]
        elif task == TaskType.LOCALIZATION_WITH_TEXT_GROUNDED:
            bboxes = result[task_prompt]["bboxes"]
            labels = result[task_prompt]["labels"]
        else:
            raise ValueError(f"Invalid task type: {task}")

        
        

        return LocalizationResp(bboxes=bboxes, labels=labels)

#%% For profiling

import cProfile
import pstats
if __name__ == "__main__":
    model = Florence2()
    image = Image.open("/Users/sarathmenon/Documents/master_thesis/datasets/resized_media/gameplay_images/hogwarts_legacy/0.jpg")


    # phrase grounded detection
    req = PredictionReq(image=image, task=TaskType.LOCALIZATION_WITH_TEXT_GROUNDED, input_text='book')
    results = model.predict(req)
    print(f"Grounded detection: {results}")

    # open vocabulary detection
    req = PredictionReq(image=image, task=TaskType.LOCALIZATION_WITH_TEXT_OPEN_VOCAB, input_text='book')
    results = model.predict(req)
    print(f"Open vocabulary detection: {results}")

    # main()
    # cProfile.run('main()', filename='profile_results.prof')
    # stats = pstats.Stats('profile_results.prof')
    # stats.sort_stats('cumulative')
    # stats.print_stats(10)  # Print the top 10 results
# %%
