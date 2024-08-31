#%%
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
from typing import List, Dict, Optional, TypedDict, Union, NamedTuple
import json
from PIL import Image
import uuid
from clicking.prompt_refinement.types import *
from clicking.common.types import ClickingImage, ObjectCategory, ImageObject

# set API keys
dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

class PromptRefiner(ImageProcessorBase):
    def __init__(self, prompt_path: str, model: str = "gpt-4o", temperature: float = 0.0):
        super().__init__(model, temperature)
        self.prompt_manager = PromptManager(prompt_path)
        self.messages = [{"role": "system", "content": self.prompt_manager.get_prompt(type='system')}]
    
    async def process_prompts_async(self, clicking_image: ClickingImage, mode: PromptMode = PromptMode.IMAGE_TO_OBJECT_DESCRIPTIONS, **kwargs) -> ClickingImage:
        description = await self._process_single_prompt(clicking_image.image, mode, **kwargs)
        
        objects = []
        for obj in description['objects']:
            # Find matching object in original clicking_image, if any
            matching_obj = next((o for o in clicking_image.predicted_objects if o.name == obj['name']), None)
            
            objects.append(ImageObject(
                name=obj['name'],
                description=obj['description'],
                category=ObjectCategory(obj['category']),
                bbox=matching_obj.bbox if matching_obj else None,
                mask=matching_obj.mask if matching_obj else None
            ))
        
        clicking_image.predicted_objects = objects
        return clicking_image

    async def _process_single_prompt(self, image: Image.Image, mode: PromptMode, object_name: Optional[str] = None, **kwargs) -> SinglePromptResponse:
        base64_image = self._pil_to_base64(image)
        template_values = self._get_template_values(mode, object_name, **kwargs)
        prompt = self.prompt_manager.get_prompt(type='user', prompt_key=mode.value, template_values=template_values)
        response = await super()._get_image_response(base64_image, prompt, self.messages, json_mode=True)

        response_dict = json.loads(response)

        if mode == PromptMode.IMAGE_TO_OBJECT_DESCRIPTIONS:
            # sort objects by category
            response_dict['objects'] = sorted(response_dict['objects'], key=lambda x: x['category'])

        return response_dict

    def _get_template_values(self, mode: PromptMode, object_name: Optional[str], **kwargs) -> TemplateValues:
        if object_name is None:
            return {}
        elif mode == PromptMode.OBJECTS_LIST_TO_DESCRIPTIONS:
            return {"input_description": object_name, "word_limit": str(kwargs.get('word_limit', 10))}
        elif mode == PromptMode.IMAGE_TO_OBJECT_DESCRIPTIONS:
            return {"description_length": kwargs.get('description_length', 20)}
        raise ValueError(f"Invalid mode: {mode}")

    def show_messages(self) -> None:
        for message in self.messages:
            print(message)

    def process_prompts(self, clicking_image: ClickingImage, mode: PromptMode = PromptMode.IMAGE_TO_OBJECT_DESCRIPTIONS, **kwargs) -> ClickingImage:
        return asyncio.run(self.process_prompts_async(clicking_image, mode, **kwargs))

# %% get expanded description from class label

if __name__ == "__main__":
    from PIL import Image
    import asyncio
    import nest_asyncio
    nest_asyncio.apply()

    image = Image.open("./datasets/resized_media/gameplay_images/mario_odessey/8.jpg")

    # Create an instance of PromptRefiner
    prompt_refiner = PromptRefiner(prompt_path="./prompts/prompt_refinement.md")

    # Define the batch of screenshots and corresponding input texts
    screenshots = [image, image]
    input_texts = [
        "car",
        "flag"
    ]

    # Define the mode and word limit for the prompts
    mode = PromptMode.OBJECTS_LIST_TO_DESCRIPTIONS
    word_limit = 5

    # Call process_prompts asynchronously
    async def process_batch_prompts():
        results = await prompt_refiner.process_prompts_async(screenshots, mode, input_texts=input_texts, word_limit=word_limit)
        print(results)

    # Run the asynchronous function
    asyncio.get_event_loop().run_until_complete(process_batch_prompts())

# %% get class labels from image

if __name__ == "__main__":
    from PIL import Image
    import asyncio
    import nest_asyncio
    nest_asyncio.apply()

    image_1 = Image.open("./datasets/resized_media/gameplay_images/unpacking/1.jpg")
    image_2 = Image.open("./datasets/resized_media/gameplay_images/mario_odessey/8.jpg")

    # Create an instance of PromptRefiner
    prompt_refiner = PromptRefiner(prompt_path="./prompts/prompt_refinement.md")

    # Define the batch of screenshots 
    images = [image_1, image_2]

    # Define the mode and word limit for the prompts
    mode = PromptMode.IMAGE_TO_OBJECT_DESCRIPTIONS

      # Call process_prompts asynchronously
    async def process_batch_prompts():
        results = await prompt_refiner.process_prompts_async(images, mode)
        for result in results:
            print(result)

    # Run the asynchronous function
    asyncio.get_event_loop().run_until_complete(process_batch_prompts())

