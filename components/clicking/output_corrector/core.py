#%%
from litellm import completion, acompletion
import os
import dotenv
import base64
import io
from PIL import Image
import numpy as np
from clicking.common.mask import SegmentationMask, SegmentationMode
from clicking.prompt_manager.core import PromptManager
import asyncio
from clicking.common.image_utils import ImageProcessorBase
from clicking.vision_model.visualization import overlay_bounding_box
from clicking.common.data_structures import ClickingImage
from clicking.prompt_refinement.data_structures import *
import json
from pydantic import BaseModel, Field
from typing import Literal
from clicking.common.data_structures import Validity

# set API keys
dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

class CorrectedResponse(BaseModel):
    judgement: Literal["true", "false"] = Field(default="true")
    visibility: Literal["fully visible", "partially visible", "hidden"] = Field(default="fully visible")
    reasoning: str

class BBoxVerificationMode(str):
    OVERLAY = "bbox_overlay"
    CROP = "bbox_crop"

class OutputCorrector(ImageProcessorBase):
    def __init__(self, prompt_path: str, model: str = "gpt-4o", temperature: float = 0.0):
        super().__init__(model, temperature)
        self.PROMPT_PATH = prompt_path
        self.prompt_manager = PromptManager(self.PROMPT_PATH)
        self.system_message = {"role": "system", "content": self.prompt_manager.get_prompt(type='system')}
        self.save_path = "./datasets/resized_media/gameplay_images/unpacking"

    async def verify_bboxes_async(self, clicking_image: ClickingImage, mode: BBoxVerificationMode = BBoxVerificationMode.CROP) -> ClickingImage:
        screenshot = clicking_image.image

        print(f"Predicted objects: {clicking_image.predicted_objects}")
        
        if mode == BBoxVerificationMode.OVERLAY:
            images = [overlay_bounding_box(screenshot.copy(), obj.bbox) for obj in clicking_image.predicted_objects]
        elif mode == BBoxVerificationMode.CROP:
            images = [obj.bbox.extract_area(screenshot.copy(), padding=0) for obj in clicking_image.predicted_objects]
        else:
            raise ValueError(f"Invalid mode: {mode}")
        
        import matplotlib.pyplot as plt

        async def process_object(image, obj):
            template_values = {"object_name": obj.name}
            prompt = self.prompt_manager.get_prompt(type='user', prompt_key=mode, template_values=template_values)

            # Create a new message list for each object
            messages = [self.system_message, {"role": "user", "content": prompt}]

            response: CorrectedResponse = await self._get_image_response(image, prompt, messages, output_type=CorrectedResponse)

            if response.judgement == "false" or response.visibility != "fully visible":
                obj.validity.is_valid = False

            print(f"Object: {obj.name}, Visibility: {response.visibility}")
            obj.validity.reason = response.reasoning

            # save_path = f"{self.save_path}/{obj.name}.png"
            # image.save(save_path)

            plt.imshow(image)
            plt.axis('off')
            plt.title(f"Object: {obj.name}, Visibility: {response.visibility}")
            plt.show()

        tasks = [process_object(image, obj) for image, obj in zip(images, clicking_image.predicted_objects)]
        await asyncio.gather(*tasks)

        return clicking_image

    def verify_bboxes(self, clicking_image: ClickingImage) -> ClickingImage:
        return asyncio.run(self.verify_bboxes_async(clicking_image))

    async def verify_masks_async(self, clicking_image: ClickingImage) -> ClickingImage:
        verification_results = {}
        screenshot = clicking_image.image
        
        extracted_areas = [obj.mask.extract_area(screenshot, padding=10) for obj in clicking_image.predicted_objects]
        base64_images = [self._pil_to_base64(area) for area in extracted_areas]
        object_names = [obj.name for obj in clicking_image.predicted_objects]
        
        responses = await self._get_image_responses(base64_images, object_names)
        verification_results[clicking_image.id] = responses

        for response in responses:
            print(response)
        
        return clicking_image

    def verify_masks(self, clicking_image: ClickingImage) -> ClickingImage:
        return asyncio.run(self.verify_masks_async(clicking_image))
