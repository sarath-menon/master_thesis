from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import os
import utils

from transformers.dynamic_module_utils import get_imports
from unittest.mock import patch

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    """Work around for https://huggingface.co/microsoft/phi-1_5/discussions/72."""
    if not str(filename).endswith("/modeling_florence2.py"):
        return get_imports(filename)
    imports = get_imports(filename)
    imports.remove("flash_attn")
    return imports

class Florence2Model:    
    def __init__(self):
        self.MODEL = 'microsoft/Florence-2-large-ft'

        # self.prompts_dict = utils.markdown_to_dict(self.PROMPT_PATH)

        with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
            self.model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base-ft", trust_remote_code=True)
            self.processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base-ft", trust_remote_code=True)

    def run_example(self, image,task_prompt, text_input=None):
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
        parsed_answer = self.processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(image.width, image.height)
        )

        return parsed_answer

    
    def plot_bbox(self, image, data):
    # Create a figure and axes
        fig, ax = plt.subplots()

        # Display the image
        ax.imshow(image)

        # Plot each bounding box
        for bbox, label in zip(data['bboxes'], data['labels']):
            # Unpack the bounding box coordinates
            x1, y1, x2, y2 = bbox
            # Create a Rectangle patch
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')
            # Add the rectangle to the Axes
            ax.add_patch(rect)
            # Annotate the label
            plt.text(x1, y1, label, color='white', fontsize=8, bbox=dict(facecolor='red', alpha=0.5))

        # Remove the axis ticks and labels
        ax.axis('off')

        # Show the plot
        plt.show()
            

if __name__ == "__main__":
    model = Florence2Model()
    image = Image.open("screenshot.jpg")  

    # # detailed captioning
    # task_prompt = '<DETAILED_CAPTION>'
    # result = model.run_example(image, task_prompt)
    # print(result)

    # phrase grounded detection
    prompt = 'Locate the player'
    task_prompt = '<CAPTION_TO_PHRASE_GROUNDING>'
    results = model.run_example(image, task_prompt, text_input=prompt)
    fig = model.plot_bbox(image, results['<CAPTION_TO_PHRASE_GROUNDING>'])

