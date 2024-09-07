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
from tqdm.asyncio import tqdm
from clicking.common.data_structures import ModuleMode
from clicking.pipeline.core import PipelineState

# set API keys
dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

class CorrectedResponse(BaseModel):
    judgement: Literal["true", "false"] = Field(default="true")
    visibility: Literal["fully visible", "partially visible", "hidden"] = Field(default="fully visible")
    reasoning: str


def process_overlay(clicking_image: ClickingImage) -> List[Image.Image]:
    result = []
    for obj in clicking_image.predicted_objects:
        if obj.bbox is None:
            print(f"Warning: Skipping object {obj.name} since it has no bbox.")
            continue
        result.append(overlay_bounding_box(clicking_image.image.copy(), obj.bbox))
    return result

def process_crop(clicking_image: ClickingImage) -> List[Image.Image]:
    result = []
    for obj in clicking_image.predicted_objects:
        if obj.bbox is None:
            print(f"Warning: Skipping object {obj.name} since it has no bbox.")
            continue
        result.append(obj.bbox.extract_area(clicking_image.image.copy(), padding=0))
    return result

class BBoxVerificationMode(Enum):
    OVERLAY = ModuleMode("bbox_overlay", process_overlay)
    CROP = ModuleMode("bbox_crop", process_crop)

class OutputCorrector(ImageProcessorBase):
    def __init__(self, prompt_path: str, model: str = "gpt-4o", temperature: float = 0.0):
        super().__init__(model, temperature)
        self.PROMPT_PATH = prompt_path
        self.prompt_manager = PromptManager(self.PROMPT_PATH)
        self.system_message = {"role": "system", "content": self.prompt_manager.get_prompt(type='system')}
        self.save_path = "./datasets/resized_media/gameplay_images/unpacking"

    async def verify_bboxes_async(self, state: PipelineState, bbox_verification_mode: BBoxVerificationMode, batch_size: int = 20) -> PipelineState:
        clicking_images = state.images
        batch_delay = 10  # Delay between batches in seconds

        async for batch_start in tqdm(range(0, len(clicking_images), batch_size), total=(len(clicking_images) + batch_size - 1) // batch_size, desc="Processing images"):
            batch_end = min(batch_start + batch_size, len(clicking_images))
            batch_clicking_images = clicking_images[batch_start:batch_end]

            tasks = []
            for clicking_image in batch_clicking_images:

                object_images = bbox_verification_mode.value.handler(clicking_image)

                # tasks.extend([self._process_object(image, obj, bbox_verification_mode) 
                
                # for image, obj in zip(object_images, clicking_image.predicted_objects)])

            await asyncio.gather(*tasks)

            # Add delay between batches to respect API rate limits
            await asyncio.sleep(batch_delay)

        state.images = clicking_images
        return state

    def verify_bboxes(self, state: PipelineState, bbox_verification_mode: BBoxVerificationMode) -> PipelineState:
        return asyncio.run(self.verify_bboxes_async(state, bbox_verification_mode))

    async def _process_object(self, image, obj, mode):
        template_values = {"object_name": obj.name}
        prompt = self.prompt_manager.get_prompt(type='user', prompt_key=mode, template_values=template_values)

        messages = [self.system_message, {"role": "user", "content": prompt}]

        response: CorrectedResponse = await self._get_image_response(image, prompt, messages, output_type=CorrectedResponse)

        if response.judgement == "false" or response.visibility != "fully visible":
            obj.validity.is_valid = False

        obj.validity.reason = response.reasoning

    def verify_masks(self, clicking_image: ClickingImage) -> ClickingImage:
        return asyncio.run(self.verify_masks_async(clicking_image))
