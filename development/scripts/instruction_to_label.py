#%%
from components.clicking.prompt_manager.core import PromptManager
from PIL import Image
from matplotlib import pyplot as plt
from enum import Enum, auto

# %%
image = Image.open("./datasets/resized_media/gameplay_images/mario_odessey/8.jpg")
plt.grid(False)
plt.axis('off')
plt.imshow(image)

#%%
from litellm import completion
import io
import base64

class PromptMode(Enum):
    LABEL = auto()
    EXPANDED_DESCRIPTION = auto()

class PromptRefiner:

    
    def __init__(self, prompt_path: str, model: str = "gpt-4o" ):
        self.model = model
        self.PROMPT_PATH = prompt_path
        
        self.prompt_manager = PromptManager(self.PROMPT_PATH)

        self.messages = [
            {"role": "system", "content": self.prompt_manager.get_prompt(type='system')},
        ]
        
    
    def _get_image_response(self, base64_image: str, text_prompt: str):
        msg = {"role": "user", "content": [
                {"type": "text", "text": text_prompt},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{base64_image}"}
                }
            ]}

        self.messages.append(msg)
        response = completion(model=self.model, messages=self.messages, response_format={ "type": "json_object" })
        return response["choices"][0]["message"]["content"]
    
    def _pil_to_base64(self, image):
        with io.BytesIO() as buffer:
            image.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def process_prompt(self, screenshot: str, input_text: str, mode: PromptMode, word_limit: int = 10):
        if mode == PromptMode.LABEL:
            return self.get_label(screenshot, input_text)
        elif mode == PromptMode.EXPANDED_DESCRIPTION:
            return self.get_expanded_description(screenshot, input_text, word_limit)
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def get_label(self, screenshot: str, action: str):
        template_values = {"action": action}
        base64_image = self._pil_to_base64(screenshot)
        prompt = self.prompt_manager.get_prompt(type='user', prompt_key='default', template_values=template_values)
        return self._get_image_response(base64_image, prompt)

    def get_expanded_description(self, screenshot: str, input_description: str, word_limit: int = 10):
        template_values = {
            "input_description": input_description,
            "word_limit": str(word_limit)
        }
        base64_image = self._pil_to_base64(screenshot)
        prompt = self.prompt_manager.get_prompt(type='user', prompt_key='prompt_expansion', template_values=template_values)
        return self._get_image_response(base64_image, prompt)

    def show_messages(self):
        for message in self.messages:
            print(message)
#%%
labeller =  PromptRefiner(prompt_path="./prompts/instruction_refinement.md")

# response = labeller.get_label(image, "Pick up the flag")
response = labeller.process_prompt(image, "yellow car", PromptMode.EXPANDED_DESCRIPTION)
print(response)
# %%
import re