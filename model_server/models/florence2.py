from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import os
import numpy as np
from transformers.dynamic_module_utils import get_imports
from .clicking_model import BaseClickingModel

class Florence2Model(BaseClickingModel):
    def __init__(self, model_id='microsoft/Florence-2-base-ft'):
        self.model, self.processor = self.load_model(model_id)

    def load_model(self, model_id):
        from transformers.dynamic_module_utils import get_imports
        from unittest.mock import patch

        def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
            if not str(filename).endswith("/modeling_florence2.py"):
                return get_imports(filename)
            imports = get_imports(filename)
            imports.remove("flash_attn")
            return imports

        with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
            model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
            processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        return model, processor

    def run_inference(self, image, task_prompt, text_input=None):
        if text_input is None:
            prompt = task_prompt
        else:
            prompt = task_prompt + text_input
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        generated_ids = self.model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
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

        response = {
            "bboxes": result[task_prompt]["bboxes"],
            "labels": result[task_prompt]["labels"]
        }
        return response

## For profiling

# def main():
#     model = Florence2Model()
#     image = Image.open("/Users/sarathmenon/Documents/master_thesis/datasets/resized_media/gameplay_images/hogwarts_legacy/0.jpg")


#     # phrase grounded detection
#     prompt = 'sword.'
#     task_prompt = '<CAPTION_TO_PHRASE_GROUNDING>'
#     results = model.run_inference(image, task_prompt, text_input=prompt)
#     print(results)

# import cProfile
# import pstats
# if __name__ == "__main__":
#     # main()
#     cProfile.run('main()', filename='profile_results.prof')
#     stats = pstats.Stats('profile_results.prof')
#     stats.sort_stats('cumulative')
#     stats.print_stats(10)  # Print the top 10 results