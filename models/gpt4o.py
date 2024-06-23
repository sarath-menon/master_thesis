import os
from openai import AsyncOpenAI
import asyncio
from . import utils


class GPT4OModel:    

    def __init__(self):
        self.MODEL = "gpt-4o"
        self.PROMPT_PATH = 'prompts/gpt4-0.md'

        self.client = AsyncOpenAI(
            # This is the default and can be omitted
            api_key=os.environ.get("OPENAI_API_KEY"),
        )
        self.prompts_dict = utils.markdown_to_dict(self.PROMPT_PATH)

    async def generate_response(self,  base64_image):

        stream = await self.client.chat.completions.create(
            model=self.MODEL,
            messages=[
                {"role": "system", "content": self.prompts_dict['System']},
                {"role": "user", "content": [
                    {"type": "text", "text": self.prompts_dict['User']},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"}
                    }
                ]}
            ],
            stream=True,
            temperature=0.0,
        )

        return stream
