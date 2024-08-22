#%%

from litellm import completion, acompletion
import os
import dotenv
import base64
import io
from PIL import Image
from clicking.visualization.mask import SegmentationMask, SegmentationMode
import asyncio

# set API keys
dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

class OutputCorrector:
    def __init__(self, model: str = "gpt-4o", temperature: float = 0.0):
        self.model = model
        self.temperature = temperature
        self.messages = [
            {"role": "system", "content": "You are a helpful assistant and in an annotating videogame images."},
        ]

    def _get_prompt(self, class_label: str):
        prompt = f"""
        Evaluate if the object in the bounding box is a {class_label}. Choose one:
        1. Class label is correct: Return the same class label.
        2. Class label is slightly off: Provide the correct class label.
        3. Class label is completely wrong: Provide the correct class label.

        Return JSON:
        {{
            "class_label": "correct or updated label",
            "judgement": "correct|slightly_off|wrong",
            "reasoning": "Brief 10-word explanation"
        }}
        """
        return prompt

    def _pil_to_base64(self, image):
        with io.BytesIO() as buffer:
            image.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode("utf-8")

    async def _get_text_response(self, prompt: str):
        prompt = self._get_prompt(prompt)
        self.add_message(prompt)
        response = await acompletion(model=self.model, messages=self.messages, response_format={"type": "json_object"}, temperature=self.temperature)
        return response["choices"][0]["message"]["content"]

    async def _get_image_responses(self, base64_images: list, text_prompts: list):
        tasks = []
        for base64_image, text_prompt in zip(base64_images, text_prompts):
            msg = {"role": "user", "content": [
                    {"type": "text", "text": text_prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"}
                    }
                ]}
            # Create a copy of messages for each task
            messages_copy = self.messages.copy()
            messages_copy.append(msg)
            task = acompletion(model=self.model, messages=messages_copy)
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks)
        return [resp["choices"][0]["message"]["content"] for resp in responses]

    async def verify_bboxes(self, screenshots: list[Image.Image], class_labels: list[str]):
        base64_images = [self._pil_to_base64(screenshot) for screenshot in screenshots]
        text_prompts = [self._get_prompt(class_label) for class_label in class_labels]
        return await self._get_image_responses(base64_images, text_prompts)

    async def verify_masks(self, screenshots: list[Image.Image], masks: list[SegmentationMask], class_labels: list[str]):
        base64_images = []
        text_prompts = []
        for screenshot, mask, class_label in zip(screenshots, masks, class_labels):
            extracted_area = mask.extract_area(screenshot, padding=10)
            base64_images.append(self._pil_to_base64(extracted_area))
            text_prompts.append(self._get_prompt(class_label))
        return await self._get_image_responses(base64_images, text_prompts)

#%% Load test image

if __name__ == "__main__":
    # from clicking.visualization.core import PromptRefiner
    from PIL import ImageDraw, Image
    import matplotlib.pyplot as plt
    from clicking.visualization.core import overlay_bounding_box
    from clicking.visualization.bbox import BoundingBox, BBoxMode


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
    output_corrector = OutputCorrector()

    # Call process_prompts asynchronously
    async def process_batch_prompts():
        results = await output_corrector.verify_bboxes(images, class_labels)
        for result, class_label in zip(results, class_labels):
            print(f"Class label: {class_label}")
            print(f"Result: {result}")

    # Run the asynchronous function
    await process_batch_prompts()
# %%
