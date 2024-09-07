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
from clicking.common.data_structures import ObjectImageDict
from clicking.common.data_structures import ImageObject
# set API keys
dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
        
class ObjectValidationResult(BaseModel):
    object_name: str = Field(default="")
    object_id: str = Field(default="")
    judgement: Literal["true", "false"] = Field(default="true")
    visibility: Literal["fully visible", "partially visible", "hidden"] = Field(default="fully visible")
    reasoning: str


def process_overlay(image: Image.Image, obj: ImageObject) -> List[Image.Image]:
    return overlay_bounding_box(image, obj.bbox)

def process_crop(image: Image.Image, obj: ImageObject) -> List[Image.Image]:
    return obj.bbox.extract_area(image, padding=0)

class BBoxVerificationMode(Enum):
    OVERLAY = ModuleMode("bbox_overlay", process_overlay)
    CROP = ModuleMode("bbox_crop", process_crop)

class OutputCorrector(ImageProcessorBase):
    def __init__(self, prompt_path: str, config: Dict, model: str = "gpt-4o", temperature: float = 0.0):
        super().__init__(model, temperature)
        self.PROMPT_PATH = prompt_path
        self.prompt_manager = PromptManager(self.PROMPT_PATH)
        self.messages = [{"role": "system", "content": self.prompt_manager.get_prompt(type='system')}]

        self.config = config

    async def verify_bboxes_async(self, state: PipelineState, bbox_verification_mode: BBoxVerificationMode, batch_size: int = 20, **kwargs) -> PipelineState:
        import matplotlib.pyplot as plt
        objects = state.get_all_predicted_objects()
        batch_delay = 10  # Delay between batches in seconds

        # Process images and prepare prompts
        processed_images, prompts, messages = [], [], []
        for obj_dict in objects.values():
            clicking_img = state.get_image_by_id(obj_dict.image_id)
            if obj_dict.object.bbox is None:
                print(f"Warning: Skipping object {obj_dict.object.name} due to missing bbox.")
                continue

            processed_images.append(bbox_verification_mode.value.handler(clicking_img.image, obj_dict.object))
            template_values = self._get_template_values(bbox_verification_mode.value, obj_dict.object)
            prompts.append(self.prompt_manager.get_prompt(type='user', prompt_key=bbox_verification_mode.value.name, template_values=template_values))
            messages.append(self.messages.copy())

        total_batches = (len(processed_images) + batch_size - 1) // batch_size
        batch_results = []

        async for batch_start in tqdm  (range(0, len(processed_images), batch_size), total=total_batches, desc="Processing images"):
            batch_end = min(batch_start + batch_size, len(objects))
            batch_images = processed_images[batch_start:batch_end]
            batch_prompts = prompts[batch_start:batch_end]
            batch_messages = messages[batch_start:batch_end]

            # show images
            for image,prompt in zip(batch_images,batch_prompts):
                # print(prompt)
                plt.imshow(image)
                plt.axis('off')
                plt.show()

            batch_response = await self._get_batch_image_responses(batch_images, batch_prompts, batch_messages, ObjectValidationResult)
            batch_results.extend(batch_response)

            # Add delay between batches to respect API rate limits
            await asyncio.sleep(batch_delay)

        for response in batch_results:
            print(response)
            image_id = objects[response.object_id].image_id

            # get the obj
     
            if response.judgement == "false" or response.visibility != "fully visible":
                is_valid = False

            # = Validity(is_valid=response.judgement == "true" and response.visibility == "fully visible")


        # for image, result in zip(state.images, batch_results):
        #     image.predicted_objects = [obj for obj in result.objects]

        return state


    def verify_bboxes(self, state: PipelineState, bbox_verification_mode: BBoxVerificationMode) -> PipelineState:
        return asyncio.run(self.verify_bboxes_async(state, bbox_verification_mode))


    def _get_template_values(self, mode: PromptMode, object: ImageObject, **kwargs) -> TemplateValues:
        # word_limits = self.config['prompts']['word_limits'].get(mode.value, {})
        # description_length = word_limits.get('description_length', 20)  # Default to 20 if not specified
        # object_name_limit = word_limits.get('object_name', 5)  # Default to 5 if not specified

        return {
            "object_name": object.name,
            "object_id": object.id
        }



    def verify_masks(self, clicking_image: ClickingImage) -> ClickingImage:
        return asyncio.run(self.verify_masks_async(clicking_image))
