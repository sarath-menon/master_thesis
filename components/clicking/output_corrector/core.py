#%%

from litellm import completion
import os
import dotenv
import base64
import io
from PIL import Image
from clicking.visualization.mask import SegmentationMask, SegmentationMode

# set API keys
dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

class OutputCorrector:
    def __init__(self, model: str = "gpt-4o"):
        self.model = model
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
    
    def _get_text_response(self, prompt: str):
        prompt = self._get_prompt(prompt)
        self.add_message(prompt)
        response = completion(model=self.model, messages=self.messages)
        return response["choices"][0]["message"]["content"]
    
    # def _get_image_response(self, base64_image: str, text_prompt: str):
    #     msg = {"role": "user", "content": [
    #             {"type": "text", "text": text_prompt},
    #             {"type": "image_url", "image_url": {
    #                 "url": f"data:image/png;base64,{base64_image}"}
    #             }
    #         ]}

    #     self.messages.append(msg)
    #     response = completion(model=self.model, messages=self.messages)
    #     return response["choices"][0]["message"]["content"]
    
    def _pil_to_base64(self, image):
        with io.BytesIO() as buffer:
            image.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode("utf-8")

    # def verify_bbox(self, screenshot: Image.Image, class_label: str):
    #     base64_image = self._pil_to_base64(screenshot)
    #     text_prompt = self._get_prompt(class_label)
    #     return self._get_image_response(base64_image, text_prompt)

    # def verify_mask(self, screenshot: Image.Image, mask: SegmentationMask, class_label: str):
    #     extracted_area = mask.extract_area(screenshot, padding=10)
    #     base64_image = self._pil_to_base64(extracted_area)
    #     text_prompt = self._get_prompt(class_label)
    #     return self._get_image_response(base64_image, text_prompt)

    def _get_image_responses(self, base64_images: list, text_prompts: list):
        messages = []
        for base64_image, text_prompt in zip(base64_images, text_prompts):
            msg = {"role": "user", "content": [
                    {"type": "text", "text": text_prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"}
                    }
                ]}
            messages.append(msg)

        self.messages.extend(messages)
        response = completion(model=self.model, messages=self.messages)
        return [resp["message"]["content"] for resp in response["choices"]]

    def verify_bboxes(self, screenshots: list[Image.Image], class_labels: list[str]):
        base64_images = [self._pil_to_base64(screenshot) for screenshot in screenshots]
        text_prompts = [self._get_prompt(class_label) for class_label in class_labels]
        return self._get_image_responses(base64_images, text_prompts)

    def verify_masks(self, screenshots: list[Image.Image], masks: list[SegmentationMask], class_labels: list[str]):
        base64_images = []
        text_prompts = []
        for screenshot, mask, class_label in zip(screenshots, masks, class_labels):
            extracted_area = mask.extract_area(screenshot, padding=10)
            base64_images.append(self._pil_to_base64(extracted_area))
            text_prompts.append(self._get_prompt(class_label))
        return self._get_image_responses(base64_images, text_prompts)

    def show_messages(self):
        for message in self.messages:
            print(message)

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
    for img in images_overlayed:
        plt.figure()
        plt.grid(False)
        plt.axis('off')
        plt.imshow(img)

    # Batch verify bounding boxes
    output_corrector = OutputCorrector()
    response = output_corrector.verify_bboxes(images, class_labels)
    print(response)