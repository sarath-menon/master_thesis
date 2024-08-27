#%%
from litellm import completion, acompletion
import os
import dotenv
import base64
import io
from enum import Enum, auto
from components.clicking.prompt_manager.core import PromptManager
import asyncio
import nest_asyncio
from typing import List, Dict, Any, Optional
import json

# set API keys
dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

class PromptMode(Enum):
    LABEL = "prompt_to_class_label"
    EXPANDED_DESCRIPTION = "prompt_expansion"
    IMAGE_TO_CLASS_LABEL = "image_to_class_label"

class PromptRefiner:    
    def __init__(self, prompt_path: str, model: str = "gpt-4o", temperature: float = 0.0):
        self.model = model
        self.temperature = temperature
        self.prompt_manager = PromptManager(prompt_path)
        self.messages = [{"role": "system", "content": self.prompt_manager.get_prompt(type='system')}]
        self.lock = asyncio.Lock()
    
    async def process_prompts(self, screenshots: List[str], mode: PromptMode, input_texts: Optional[List[str]] = None, **kwargs):

        # handle case where input_texts is None
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
        response = await self._get_image_response(base64_image, prompt, json_mode=mode != PromptMode.EXPANDED_DESCRIPTION)

        if mode == PromptMode.IMAGE_TO_CLASS_LABEL:
            response = json.loads(response)
        elif mode == PromptMode.EXPANDED_DESCRIPTION:
            response = {input_text: response}

         # add input image and input text to response
        response['input_image'] = screenshot

        return response

    def _get_template_values(self, mode: PromptMode, input_text: str, **kwargs) -> Dict[str, Any]:
        if input_text is None:
            return {}
        elif mode == PromptMode.LABEL:
            return {"action": input_text}
        elif mode == PromptMode.EXPANDED_DESCRIPTION:
            return {"input_description": input_text, "word_limit": str(kwargs.get('word_limit', 10))}
        elif mode == PromptMode.IMAGE_TO_CLASS_LABEL:
            return {"description_length": kwargs.get('description_length', 20)}
        raise ValueError(f"Invalid mode: {mode}")

    async def _get_image_response(self, base64_image: str, text_prompt: str, json_mode: bool = False):
        msg = {
            "role": "user", 
            "content": [
                {"type": "text", "text": text_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
            ]
        }

        async with self.lock: 
            self.messages.append(msg)

        response_format = {"type": "json_object"} if json_mode else None
        response = await acompletion(
            model=self.model, 
            messages=self.messages, 
            temperature=self.temperature, 
            response_format=response_format
        )
        return response["choices"][0]["message"]["content"]

    def _pil_to_base64(self, image):
        with io.BytesIO() as buffer:
            image.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode("utf-8")

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
    mode = PromptMode.EXPANDED_DESCRIPTION
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

    image = Image.open("./datasets/resized_media/gameplay_images/unpacking/1.jpg")

    # Create an instance of PromptRefiner
    prompt_refiner = PromptRefiner(prompt_path="./prompts/prompt_refinement.md")

    # Define the batch of screenshots 
    images = [image, image]

    # Define the mode and word limit for the prompts
    mode = PromptMode.IMAGE_TO_CLASS_LABEL

      # Call process_prompts asynchronously
    async def process_batch_prompts():
        results = await prompt_refiner.process_prompts(images, mode)
        for result in results:
            print(result)

    # Run the asynchronous function
    asyncio.get_event_loop().run_until_complete(process_batch_prompts())

