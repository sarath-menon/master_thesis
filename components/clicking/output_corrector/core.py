#%%

from litellm import completion, acompletion
import os
import dotenv
import base64
import io
from PIL import Image
from clicking.visualization.mask import SegmentationMask, SegmentationMode
from components.clicking.prompt_manager.core import PromptManager
import asyncio
from components.clicking.common.image_utils import ImageProcessorBase

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
            task = self._get_image_response(base64_image, prompt, self.messages)
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks)
        return responses

    async def verify_bboxes(self, screenshots: list[Image.Image], class_labels: list[str]):
        base64_images = [self._pil_to_base64(screenshot) for screenshot in screenshots]
        return await self._get_image_responses(base64_images, class_labels)

    async def verify_masks(self, screenshots: list[Image.Image], masks: list[SegmentationMask], class_labels: list[str]):
        base64_images = []
        for screenshot, mask, class_label in zip(screenshots, masks, class_labels):
            extracted_area = mask.extract_area(screenshot, padding=10)
            base64_images.append(self._pil_to_base64(extracted_area))
        return await self._get_image_responses(base64_images, class_labels)

#%% Load test image

if __name__ == "__main__":
    from PIL import Image
    import matplotlib.pyplot as plt
    from clicking.visualization.core import overlay_bounding_box
    from clicking.visualization.bbox import BoundingBox, BBoxMode
    import asyncio
    import nest_asyncio
    nest_asyncio.apply()

    image = Image.open("./datasets/resized_media/gameplay_images/mario_odessey/8.jpg")
    plt.grid(False)
    plt.axis('off')
    plt.imshow(image)

    bboxes = [
        BoundingBox((1000, 400, 200, 200), BBoxMode.XYWH),
        BoundingBox((1000, 400, 200, 200), BBoxMode.XYWH)
    ]
    class_labels = ["building", "flat"]
    

    # Load images and apply bounding boxes
    images = [image, image]
    images_overlayed = [overlay_bounding_box(img.copy(), bbox) for img, bbox in zip(images, bboxes)]

    # Display images
    plt.imshow(images_overlayed[0])
    plt.axis(False)
    plt.axis('off')

    # Batch verify bounding boxes
    output_corrector = OutputCorrector(prompt_path="./prompts/output_corrector.md")

    # Call process_prompts asynchronously
    async def process_batch_prompts():
        results = await output_corrector.verify_bboxes(images, class_labels)
        for result, class_label in zip(results, class_labels):
            print(f"Class label: {class_label}")
            print(f"Result: {result}")

    # Run the asynchronous function
    asyncio.get_event_loop().run_until_complete(process_batch_prompts())
# %%
