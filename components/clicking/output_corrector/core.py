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
from clicking.image_processor.visualization import overlay_bounding_box
from clicking.common.data_structures import ClickingImage, ValidityStatus
from clicking.prompt_refinement.data_structures import *
import json
from pydantic import BaseModel, Field
from typing import Literal
from clicking.common.data_structures import ObjectValidity
from tqdm.asyncio import tqdm
from clicking.common.data_structures import ModuleMode
from clicking.common.data_structures import PipelineState
from clicking.common.data_structures import ObjectImageDict
from clicking.common.data_structures import ImageObject
import matplotlib.pyplot as plt
from tqdm.asyncio import tqdm as async_tqdm

# set API keys
dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
        
class ObjectValidationResult(BaseModel):
    object_name: str = Field(default="")
    object_id: str = Field(default="")
    accuracy: Literal["true", "false"] = Field(default="true")
    visibility: Literal["fully visible", "partially visible", "hidden"] = Field(default="fully visible")
    reasoning: str


def process_overlay(image: Image.Image, obj: ImageObject) -> List[Image.Image]:
    return overlay_bounding_box(image, obj.bbox, padding=10)

def process_crop(image: Image.Image, obj: ImageObject) -> List[Image.Image]:
    return obj.bbox.extract_area(image, padding=10)

class BBoxVerificationMode(Enum):
    OVERLAY = ModuleMode("bbox_overlay", process_overlay)
    CROP = ModuleMode("bbox_crop", process_crop)

class OutputCorrector(ImageProcessorBase):
    def __init__(self, config: Dict, model: str = "gpt-4o", temperature: float = 0.0):
        super().__init__(model, temperature)
        self.PROMPT_PATH = config['prompts']['output_corrector_path']
        self.prompt_manager = PromptManager(self.PROMPT_PATH)
        self.messages = [{"role": "system", "content": self.prompt_manager.get_prompt(type='system')}]

        self.config = config

    async def verify_bboxes_async(self, state: PipelineState, bbox_verification_mode: BBoxVerificationMode, batch_size: int = 20, show_images: bool = False, **kwargs) -> PipelineState:
        
        objects = state.get_all_predicted_objects()
        batch_delay = 10  # Delay between batches in seconds

        # Process images and prepare prompts
        processed_images, prompts, messages = [], [], []
        object_names = []

        for obj_dict in async_tqdm(objects.values(), desc="Processing objects"):
            if obj_dict.object.validity.status == ValidityStatus.INVALID:
                print(f"Warning: Skipping bbox verification for object {obj_dict.object.name} due to invalid bbox.")
                continue

            clicking_img = state.get_image_by_id(obj_dict.image_id)
            object_names.append(obj_dict.object.name)

            processed_images.append(bbox_verification_mode.value.handler(clicking_img.image, obj_dict.object))
            template_values = self._get_template_values(bbox_verification_mode.value, obj_dict.object)
            prompts.append(self.prompt_manager.get_prompt(type='user', prompt_key=bbox_verification_mode.value.name, template_values=template_values))
            messages.append(self.messages.copy())

        total_batches = (len(processed_images) + batch_size - 1) // batch_size
        batch_results = []

        async for batch_start in tqdm(range(0, len(processed_images), batch_size), total=total_batches, desc="Processing images"):
            batch_end = min(batch_start + batch_size, len(objects))
            batch_images = processed_images[batch_start:batch_end]
            batch_prompts = prompts[batch_start:batch_end]
            batch_messages = messages[batch_start:batch_end]

            batch_response = await self._get_batch_image_responses(batch_images, batch_prompts, batch_messages, ObjectValidationResult)
            batch_results.extend(batch_response)

            # Add delay between batches to respect API rate limits, but not after the last batch
            if batch_end < len(processed_images):
                await asyncio.sleep(batch_delay)

        # show images
        if show_images:
            for img, obj_name in zip(processed_images, object_names):
                plt.imshow(img)
                plt.title(f"Object: {obj_name}")
                plt.axis('off')
                plt.show()

        for response in batch_results:
            status = ValidityStatus.VALID
            if response.accuracy == "false" or response.visibility != "fully visible":
                status = ValidityStatus.INVALID

            obj = objects[response.object_id].object
            obj.validity = ObjectValidity(status=status,
                                    reason=response.reasoning,
                                    accuracy=response.accuracy,
                                    visibility=response.visibility)
        return state

    def verify_bboxes(self, state: PipelineState, bbox_verification_mode: BBoxVerificationMode,show_images: bool = True, **kwargs) -> PipelineState:
        return asyncio.run(self.verify_bboxes_async(state, bbox_verification_mode, show_images=show_images, **kwargs))


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
