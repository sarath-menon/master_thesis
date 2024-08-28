#%%
from components.clicking.common.image_utils import ImageProcessorBase
from litellm import completion, acompletion
import os
import dotenv
import base64
import io
from enum import Enum, auto
from clicking.prompt_manager.core import PromptManager
import asyncio
import nest_asyncio
from typing import List, Dict, Any, Optional
import json

# set API keys
dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

class PromptMode(Enum):
    OBJECTS_LIST_TO_DESCRIPTIONS = "OBJECTS_LIST_TO_DESCRIPTIONS"
    IMAGE_TO_OBJECT_DESCRIPTIONS = "IMAGE_TO_OBJECT_DESCRIPTIONS"
    IMAGE_TO_OBJECTS_LIST = "IMAGE_TO_OBJECTS_LIST"

class PromptRefiner(ImageProcessorBase):
    def __init__(self, prompt_path: str, model: str = "gpt-4o", temperature: float = 0.0):
        super().__init__(model, temperature)
        self.prompt_manager = PromptManager(prompt_path)
        self.messages = [{"role": "system", "content": self.prompt_manager.get_prompt(type='system')}]
    
    async def process_prompts(self, screenshots: List[str], mode: PromptMode, input_texts: Optional[List[str]] = None, **kwargs):
        if input_texts is None:
            input_texts = [None] * len(screenshots)

        tasks = [self._process_single_prompt(screenshot, mode, input_text, **kwargs) 
                 for screenshot, input_text in zip(screenshots, input_texts)]
        results = await asyncio.gather(*tasks)
        return results

    async def _process_single_prompt(self, screenshot: str, mode: PromptMode, input_text: Optional[str] = None, **kwargs):
        base64_image = self._pil_to_base64(screenshot)
        template_values = self._get_template_values(mode, input_text, **kwargs)

        prompt = self.prompt_manager.get_prompt(type='user', prompt_key=mode.value, template_values=template_values)

        response = await super()._get_image_response(base64_image, prompt, self.messages, json_mode=True)

        try:
            response = json.loads(response)
        except json.JSONDecodeError as e:
            return {"Failed to parse JSON response": {e}, "raw_response": response}

        if mode == PromptMode.IMAGE_TO_OBJECT_DESCRIPTIONS:
            # sort objects by category
            response['objects'] = sorted(response['objects'], key=lambda x: x['category'])

        return response

    def _get_template_values(self, mode: PromptMode, input_text: str, **kwargs) -> Dict[str, Any]:
        if input_text is None:
            return {}
        elif mode == PromptMode.OBJECTS_LIST_TO_DESCRIPTIONS:
            return {"input_description": input_text, "word_limit": str(kwargs.get('word_limit', 10))}
        elif mode == PromptMode.IMAGE_TO_OBJECT_DESCRIPTIONS:
            return {"description_length": kwargs.get('description_length', 20)}
        raise ValueError(f"Invalid mode: {mode}")

    def show_messages(self):
        for message in self.messages:
            print(message)
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
        results = await prompt_refiner.process_prompts(screenshots, mode, input_texts=input_texts, word_limit=word_limit)
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
        results = await prompt_refiner.process_prompts(images, mode)
        for result in results:
            print(result)

    # Run the asynchronous function
    asyncio.get_event_loop().run_until_complete(process_batch_prompts())

