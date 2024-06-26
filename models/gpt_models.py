import os
from openai import AsyncOpenAI
import asyncio
from . import utils
import json

class GPTModels:    
    def __init__(self, system_prompt):
        self.MODEL = "gpt-4o"
        self.client = AsyncOpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
        )
        self.chat_history = [{"role": "system", "content": system_prompt}]

    async def single_img_response_async(self, base64_image, user_prompt):
        current_msg = {"role": "user", "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"}
                    }
                ]}

        self.chat_history.append(current_msg)

        stream = await self.client.chat.completions.create(
            model=self.MODEL,
            messages=self.chat_history,
            stream=True,
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        return stream

    async def generate_waste_async(self, user_prompt):
        self.chat_history.append({"role": "user", "content": user_prompt})

        stream = await self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=self.chat_history,
            stream=True,
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        return stream
        # async for chunk in stream:
        #     yield chunk.choices[0].delta.content or ""

    def add_response_to_history(self, response):
        self.chat_history.append({"role": "assistant", "content": response})

