from clicking.common.image_utils import ImageProcessorBase
from litellm import completion, acompletion
import os
import dotenv
import base64
import io
from enum import Enum, auto
from clicking.prompt_manager.core import PromptManager
import asyncio
import nest_asyncio
import yaml
from typing import List, Dict, Optional, TypedDict, Union, NamedTuple, Type, Any
from pydantic import BaseModel
import json
from PIL import Image
import uuid
from clicking.prompt_refinement.data_structures import *
from clicking.common.data_structures import ClickingImage, ObjectCategory, ImageObject, PipelineState
from pydantic import Field
from tqdm.asyncio import tqdm

# set API keys
dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

class PromptResponse(BaseModel):
    objects: list[ImageObject]

class PromptRefiner(ImageProcessorBase):
    def __init__(self,  config: Dict, model: str = "gpt-4o", temperature: float = 0.0):
        super().__init__(model, temperature)
        self.prompt_manager = PromptManager(config['prompts']['refinement_path'])
        self.messages = [{"role": "system", "content": self.prompt_manager.get_prompt(type='system')}]
        
        # Load configuration
        self.config = config
        
    async def process_prompts_async(self, state: PipelineState, mode: PromptMode = PromptMode.IMAGE_TO_OBJECT_DESCRIPTIONS, **kwargs) -> PipelineState:
        images = [ci.image for ci in state.images]
        template_values = [self._get_template_values(mode, None, **kwargs) for _ in state.images]

        prompts = [self.prompt_manager.get_prompt(type='user', prompt_key=mode.value, template_values=tv) for tv in template_values]
        messages = [self.messages.copy() for _ in state.images] 

        batch_results = []
        batch_size = 20  # Adjust this value based on your needs and API limits
        batch_delay = 10  # Delay between batches in seconds

        total_batches = (len(images) + batch_size - 1) // batch_size

        async for batch_start in tqdm(range(0, len(images), batch_size), total=total_batches, desc="Processing images"):
            batch_end = min(batch_start + batch_size, len(images))
            batch_images = images[batch_start:batch_end]
            batch_prompts = prompts[batch_start:batch_end]
            batch_messages = messages[batch_start:batch_end]

            batch_response = await self._get_batch_image_responses(batch_images, batch_prompts, batch_messages, PromptResponse)
            batch_results.extend(batch_response)

            # Add delay between batches to respect API rate limits
            if batch_end < len(images):
                await asyncio.sleep(batch_delay)

        for clicking_image, result in zip(state.images, batch_results):
            clicking_image.predicted_objects = [obj for obj in result.objects]
        return state

    async def _process_single_image(self, clicking_image: ClickingImage, mode: PromptMode, **kwargs) -> ClickingImage:
        objects = await self._process_single_prompt(clicking_image.image, mode, **kwargs)
        
        clicking_image.predicted_objects = [obj for obj in objects if obj.category == ObjectCategory.GAME_ASSET]
        return clicking_image

    async def _process_single_prompt(self, image: Image.Image, mode: PromptMode, object_name: Optional[str] = None, **kwargs) -> List[ImageObject]:
        template_values = self._get_template_values(mode, object_name, **kwargs)
        prompt = self.prompt_manager.get_prompt(type='user', prompt_key=mode.value, template_values=template_values)
        
        try:
            response = await super()._get_image_response(image, prompt, self.messages, PromptResponse)
        except ValueError as e:
            print(f"Error processing prompt: {e}")
            return []

        if mode == PromptMode.IMAGE_TO_OBJECT_DESCRIPTIONS:
            response.objects.sort(key=lambda x: x.category)

        return response.objects

    def _get_template_values(self, mode: PromptMode, object_name: Optional[str], **kwargs) -> TemplateValues:
        word_limits = self.config['prompts']['word_limits'].get(mode.value, {})
        description_length = word_limits.get('description_length', 20)  # Default to 20 if not specified
        object_name_limit = word_limits.get('object_name', 5)  # Default to 5 if not specified
        
        if mode == PromptMode.OBJECTS_LIST_TO_DESCRIPTIONS:
            return {
                "input_description": object_name,
                "description_length": description_length,
                "object_name_limit": object_name_limit
            }
        elif mode == PromptMode.IMAGE_TO_OBJECT_DESCRIPTIONS:
            return {
                "description_length": description_length,
                "object_name_limit": object_name_limit
            }
        raise ValueError(f"Invalid mode: {mode}")

    def show_messages(self) -> None:
        for message in self.messages:
            print(message)

    def process_prompts(self, state: PipelineState, mode: PromptMode = PromptMode.IMAGE_TO_OBJECT_DESCRIPTIONS, **kwargs) -> PipelineState:
        return asyncio.run(self.process_prompts_async(state, mode, **kwargs))
