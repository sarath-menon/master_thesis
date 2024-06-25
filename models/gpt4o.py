import os
from openai import AsyncOpenAI
import asyncio
from . import utils
import json

class GPT4OModel:    
    def __init__(self, prompt_path):
        self.MODEL = "gpt-4o"
        self.PROMPT_PATH = prompt_path
        self.client = AsyncOpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
        )
        self.prompts_dict = utils.markdown_to_dict(self.PROMPT_PATH)

    async def generate_response(self, base64_image):
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
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        return stream

    async def generate_response_async(self, base64_image):
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
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        return stream

    async def generate_waste(self, base64_image):
        response = await self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a waste classification model"},
                {"role": "user", "content": "return a short json with 3 fields, one called reason"}
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        return response

    async def generate_waste_async(self, text_input):
        stream = await self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert videogame QA tester"},
                {"role": "user", "content": text_input + " and return a json with the following field 'response'"}
            ],
            stream=True,
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        return stream
        # async for chunk in stream:
        #     yield chunk.choices[0].delta.content or ""

