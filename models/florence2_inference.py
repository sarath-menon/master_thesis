from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import os
import numpy as np
import cv2

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

        # Remove the white border
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        ax.margins(0)
        ax.set_axis_off()

        # Display the image
        ax.imshow(image)

        # Plot each bounding box
        for bbox, label in zip(data['bboxes'], data['labels']):
            # Unpack the bounding box coordinates
            x1, y1, x2, y2 = bbox

            # Create a Rectangle patch
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
            # Add the rectangle to the Axes
            ax.add_patch(rect)
            # Annotate the label with an offset
            text_x = x1  
            text_y = y1 - 40  
            plt.text(text_x, text_y, label, color='white', fontsize=8, bbox=dict(facecolor='red', alpha=0.5))

        # Remove the axis ticks and labels
        ax.axis('off')

        # Adjust the figure size to match the image aspect ratio
        fig.set_size_inches(image.width / 200.0, image.height / 200.0)

        # Show the plot
        plt.show()
            

    def get_bbox_image(self, image, data):
        # Convert the image to a numpy array
        image_np = np.array(image)

        # Plot each bounding box
        for bbox, label in zip(data['bboxes'], data['labels']):
            # Unpack the bounding box coordinates and convert to integers
            x1, y1, x2, y2 = map(int, bbox)

            # Draw the rectangle on the image
            image_np = cv2.rectangle(image_np, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Annotate the label below the rectangle
            text_x = x1  
            text_y = y2 + 10  # Adjust y coordinate to be below the rectangle
            image_np = cv2.putText(image_np, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
        return image_np

        return image_np
            

if __name__ == "__main__":
    model = Florence2Model()
    image = Image.open("datasets/game_dataset/raw/free_fire/1.png").convert("RGB")

    # # detailed captioning
    # task_prompt = '<DETAILED_CAPTION>'
    # result = model.run_example(image, task_prompt)
    # print(result)

    # phrase grounded detection
    prompt = 'Locate the sword'
    task_prompt = '<CAPTION_TO_PHRASE_GROUNDING>'
    results = model.run_example(image, task_prompt, text_input=prompt)
    # fig = model.plot_bbox(image, results['<CAPTION_TO_PHRASE_GROUNDING>'])
    
    bbox_image = model.get_bbox_image(image, results['<CAPTION_TO_PHRASE_GROUNDING>'])
    plt.imshow(bbox_image)
    plt.axis('off')
    plt.show()
