from typing import Dict, List
from clicking_client import Client
from clicking.common.data_structures import *
from enum import Enum
from clicking.vision_model.utils import pil_to_base64
from clicking_client.types import Response
from clicking.common.image_utils import ImageProcessorBase
import asyncio
from tqdm.asyncio import tqdm as async_tqdm
from clicking.prompt_manager.core import PromptManager
from pydantic import BaseModel, Field
from typing import Literal
from tqdm import tqdm
        
class ClickingResult(BaseModel):
    object_id: str = Field(default="")
    reasoning: str = Field(default="")
    x: int = Field(default=0)
    y: int = Field(default=0)

def create_centered_image(input_image: Image.Image, canvas_width: int = 1024, canvas_height: int = 768):
    background = Image.new('RGBA', (canvas_width, canvas_height), (0, 0, 0, 0))
    
    scale_factor = 4
    if input_image.width > canvas_width or input_image.height > canvas_height:
        input_image = input_image.resize((input_image.width // scale_factor, input_image.height // scale_factor))
    
    x_offset = (canvas_width - input_image.width) // 2
    y_offset = (canvas_height - input_image.height) // 2
    
    result = background.copy()
    result.paste(input_image, (x_offset, y_offset))
    
    return result, (x_offset, y_offset), scale_factor

class AnthropicClicking(ImageProcessorBase):
    def __init__(self, client: Client, config: Dict, model: str = "claude-3-5-sonnet-20241022"):
        super().__init__(model)
        self.client = client
        self.config = config
        self.prompt_manager = PromptManager(config['prompts']['anthropic_clicking_path'])
        self.messages = [{"role": "system", "content": '\n' + self.prompt_manager.get_prompt(type='system')}]
        self.image_cache = {}

    async def prepare_batch_data(self, objects: Dict, state: PipelineState):
        images, prompts, messages = [], [], []
        object_names = []

        for obj_dict in async_tqdm(objects.values(), desc="Preparing images"):
            clicking_img = state.get_image_by_id(obj_dict.image_id)
            object_names.append(obj_dict.object.name)

            centered_image, offset, scale_factor = create_centered_image(clicking_img.image)
            self.image_cache[obj_dict.object.id] = (offset, scale_factor)

            template_values = {"object_name": obj_dict.object.name, "object_id": obj_dict.object.id}

            prompt = self.prompt_manager.get_prompt(type='user', prompt_key='default', template_values=template_values)
            
            # add newline to the beginning and end of the the prompt if it doesn't already have them
            if not prompt.startswith("\n"):
                prompt = "\n" + prompt
            if not prompt.endswith("\n"):
                prompt = prompt + "\n"
            
            images.append(centered_image)
            prompts.append(prompt)
            messages.append(self.messages.copy())

        return images, prompts, messages, object_names

    async def process_batch_results(self, batch_results, images, object_names, state: PipelineState):
        for i, resp in enumerate(tqdm(batch_results, desc="Postprocessing responses")):
            obj = state.find_object_by_id(resp.object_id)
            if not obj:
                print(f"Object {resp.object_id} not found in state")
                continue

            obj.clickpoint = None
            offset, scale_factor = self.image_cache[obj.id]

            print(f"{object_names[i]}: {resp.x}, {resp.y}")
            x = (resp.x - offset[0]) * scale_factor
            y = (resp.y - offset[1]) * scale_factor

            obj.clickpoint = ClickPoint(
                x=x, 
                y=y, 
                validity=ClickPointValidity(status=ValidityStatus.VALID, reason=resp.reasoning)
            )

    async def get_results_async(self, state: PipelineState, AnthropicClicking_mode: TaskType, batch_size: int = 20, batch_delay: int = 5) -> PipelineState:
        objects = state.get_all_predicted_objects()
        images, prompts, messages, object_names = await self.prepare_batch_data(objects, state)

        batch_results = []
        for batch_start in range(0, len(images), batch_size):
            batch_end = min(batch_start + batch_size, len(images))
            # Uncomment and implement _get_batch_image_responses when ready
            batch_response = await self._get_batch_image_responses(
                images[batch_start:batch_end],
                prompts[batch_start:batch_end],
                messages[batch_start:batch_end],
                ClickingResult
            )
            batch_results.extend(batch_response)

            print(f"Batch images: {images[batch_start:batch_end]}")
            print(f"Batch prompts: {prompts[batch_start:batch_end]}")
            print(f"Batch messages: {messages[batch_start:batch_end]}")

            if batch_end < len(images):
                await asyncio.sleep(batch_delay)

        await self.process_batch_results(batch_results, images, object_names, state)
        return state

    def get_results(self, state: PipelineState, pointing_mode: TaskType) -> PipelineState:
        return asyncio.run(self.get_results_async(state, pointing_mode))
