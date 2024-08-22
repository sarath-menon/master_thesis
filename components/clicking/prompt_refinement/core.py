#%%
from litellm import completion, acompletion
import os
import dotenv
import base64
import io
from enum import Enum, auto
from components.clicking.prompt_manager.core import PromptManager
import asyncio
from typing import List

# set API keys
dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

class PromptMode(Enum):
    LABEL = auto()
    EXPANDED_DESCRIPTION = auto()

class PromptRefiner:    
    def __init__(self, prompt_path: str, model: str = "gpt-4o", temperature: float = 0.0):
        self.model = model
        self.temperature = temperature
        self.PROMPT_PATH = prompt_path
        
        self.prompt_manager = PromptManager(self.PROMPT_PATH)

        self.messages = [
            {"role": "system", "content": self.prompt_manager.get_prompt(type='system')},
        ]

    async def _get_image_response(self, base64_image: str, text_prompt: str):
        msg = {"role": "user", "content": [
                {"type": "text", "text": text_prompt},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{base64_image}"}
                }
            ]}

        self.messages.append(msg)
        response = await acompletion(model=self.model, messages=self.messages, response_format={ "type": "json_object" }, temperature=self.temperature)
        return response["choices"][0]["message"]["content"]
    
    def _pil_to_base64(self, image):
        with io.BytesIO() as buffer:
            image.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode("utf-8")

    async def process_prompts(self, screenshots: List[str], input_texts: List[str], mode: PromptMode, word_limit: int = 10):
        tasks = []
        for screenshot, input_text in zip(screenshots, input_texts):
            task = self.process_prompt(screenshot, input_text, mode, word_limit)
            tasks.append(task)
        results = await asyncio.gather(*tasks)
        return results

    async def process_prompt(self, screenshot: str, input_text: str, mode: PromptMode, word_limit: int = 10):
        if mode == PromptMode.LABEL:
            return await self.get_label(screenshot, input_text)
        elif mode == PromptMode.EXPANDED_DESCRIPTION:
            return await self.get_expanded_description(screenshot, input_text, word_limit)
        else:
            raise ValueError(f"Invalid mode: {mode}")

    async def get_label(self, screenshot: str, action: str):
        template_values = {"action": action}
        base64_image = self._pil_to_base64(screenshot)
        prompt = self.prompt_manager.get_prompt(type='user', prompt_key='prompt_to_class_label', template_values=template_values)
        return await self._get_image_response(base64_image, prompt)

    async def get_expanded_description(self, screenshot: str, input_description: str, word_limit: int = 10):
        template_values = {
            "input_description": input_description,
            "word_limit": str(word_limit)
        }
        base64_image = self._pil_to_base64(screenshot)
        prompt = self.prompt_manager.get_prompt(type='user', prompt_key='prompt_expansion', template_values=template_values)
        return await self._get_image_response(base64_image, prompt)

    def show_messages(self):
        for message in self.messages:
            print(message)
# %%

# ## Sample code

# from PIL import Image
# from matplotlib import pyplot as plt

# image = Image.open("./datasets/resized_media/gameplay_images/mario_odessey/8.jpg")

# # Create an instance of PromptRefiner
# prompt_refiner = PromptRefiner(prompt_path="./prompts/prompt_refinement.md")

# # Define the batch of screenshots and corresponding input texts
# screenshots = [image, image]
# input_texts = [
#     "car",
#     "flag"
# ]

# # Define the mode and word limit for the prompts
# mode = PromptMode.EXPANDED_DESCRIPTION
# word_limit = 15

# # Call process_prompts asynchronously
# async def process_batch_prompts():
#     results = await prompt_refiner.process_prompts(screenshots, input_texts, mode, word_limit)
#     for result in results:
#         print(result)

# # Run the asynchronous function
# await process_batch_prompts()

