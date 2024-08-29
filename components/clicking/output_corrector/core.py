#%%

from litellm import completion, acompletion
import os
import dotenv
import base64
import io
from PIL import Image
import numpy as np
from clicking.vision_model.mask import SegmentationMask, SegmentationMode
from components.clicking.prompt_manager.core import PromptManager
import asyncio
from components.clicking.common.image_utils import ImageProcessorBase
from components.clicking.vision_model.visualization import overlay_bounding_box
from components.clicking.vision_model.core import LocalizationResults, SegmentationResults

# set API keys
dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

class OutputCorrector(ImageProcessorBase):
    def __init__(self, prompt_path: str, model: str = "gpt-4o", temperature: float = 0.0):
        super().__init__(model, temperature)
        self.PROMPT_PATH = prompt_path
        self.prompt_manager = PromptManager(self.PROMPT_PATH)
        self.messages = [
            {"role": "system", "content": self.prompt_manager.get_prompt(type='system')},
        ]

    async def _get_image_responses(self, base64_images: list, class_labels: list):
        tasks = []

        for base64_image, class_label in zip(base64_images, class_labels):
            template_values = {"class_label": class_label}
            prompt = self.prompt_manager.get_prompt(type='user', prompt_key='correct_class_label', template_values=template_values)
            task = self._get_image_response(base64_image, prompt, self.messages, json_mode=True)
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks)
        return responses

    async def verify_bboxes_async(self, localization_results: LocalizationResults) -> LocalizationResults:
        verification_results = {}
        for sample in localization_results.processed_samples:
            image_id = sample.image_id
            screenshot = sample.image
            bboxes = localization_results.predictions[image_id]
            descriptions = [bbox.description for bbox in bboxes]
            
            images_overlayed = [overlay_bounding_box(screenshot.copy(), bbox) for bbox in bboxes]
            base64_images = [self._pil_to_base64(img) for img in images_overlayed]
            
            responses = await self._get_image_responses(base64_images, descriptions)
            verification_results[image_id] = responses
        
        return localization_results

    def verify_bboxes(self, localization_results: LocalizationResults) -> LocalizationResults:
        return asyncio.run(self.verify_bboxes_async(localization_results))

    async def verify_masks_async(self, localization_results: SegmentationResults) -> SegmentationResults:
        verification_results = {}
        for sample in localization_results.processed_samples:
            image_id = sample.image_id
            screenshot = sample.image
            masks = localization_results.predictions[image_id]
            class_labels = [mask.object_name for mask in masks]
            
            extracted_areas = [mask.extract_area(screenshot, padding=10) for mask in masks]
            base64_images = [self._pil_to_base64(area) for area in extracted_areas]
            
            responses = await self._get_image_responses(base64_images, class_labels)
            verification_results[image_id] = responses
        
        return verification_results

    def verify_masks(self, localization_results: SegmentationResults) -> SegmentationResults:
        return asyncio.run(self.verify_masks_async(localization_results))

#%% Demo

if __name__ == "__main__":
    from PIL import Image
    import matplotlib.pyplot as plt
    from clicking.vision_model.visualization import overlay_bounding_box
    from clicking.vision_model.bbox import BoundingBox, BBoxMode
    import asyncio
    import nest_asyncio
    nest_asyncio.apply()

    image = Image.open("./datasets/resized_media/gameplay_images/mario_odessey/8.jpg")
    plt.grid(False)
    plt.axis('off')
    plt.imshow(image)

    # Batch verify bounding boxes
    output_corrector = OutputCorrector(prompt_path="./prompts/output_corrector.md")

    
    #%% Verify bounding boxes
    bboxes = [
        BoundingBox((1000, 400, 200, 200), BBoxMode.XYWH),
        BoundingBox((1000, 400, 200, 200), BBoxMode.XYWH)
    ]
    class_labels = ["building", "flat"]

    # Load images and apply bounding boxes
    images = [image, image]
    images_overlayed = [overlay_bounding_box(img.copy(), bbox) for img, bbox in zip(images, bboxes)]

    # Call process_prompts asynchronously
    async def process_batch_prompts():
        results = await output_corrector.verify_bboxes_async(images, class_labels)
        for result, class_label in zip(results, class_labels):
            print(f"Class label: {class_label}")
            print(f"Result: {result}")

    # Run the asynchronous function
    asyncio.get_event_loop().run_until_complete(process_batch_prompts())

    #%% Verify masks
    masks = []
    mask_class_labels = ["flag", "building"]

    for _ in range(2):
        mask_array = np.zeros((image.height, image.width), dtype=bool)
        rect_width, rect_height = 120, 100
        x_start, y_start = 1020, 440
        mask_array[y_start:y_start+rect_height, x_start:x_start+rect_width] = True
        masks.append(SegmentationMask(mask_array, mode=SegmentationMode.BINARY_MASK))

    # Batch verify masks
    async def process_batch_masks():
        results = await output_corrector.verify_masks_async(images, masks, mask_class_labels)
        for result, class_label in zip(results, mask_class_labels):
            print(f"Mask class label: {class_label}")
            print(f"Mask verification result: {result}")

    # Run the asynchronous function for mask verification
    asyncio.get_event_loop().run_until_complete(process_batch_masks())


    fig, axs = plt.subplots(len(images), len(masks) + 1, figsize=(18, 12))

    for row, (img, img_label) in enumerate(zip(images, class_labels)):
        # Display original images in the first column
        axs[row, 0].imshow(img)
        axs[row, 0].set_title(f'Original Image ({img_label})')
        axs[row, 0].axis('off')

        # Display extracted areas in the subsequent columns
        for col, (mask, class_label) in enumerate(zip(masks, mask_class_labels)):
            extracted_area = mask.extract_area(img, padding=10)
            axs[row, col + 1].imshow(extracted_area)
            axs[row, col + 1].set_title(f'Extracted Area ({class_label})')
            axs[row, col + 1].axis('off')

    plt.tight_layout()
    plt.show()

# %%
