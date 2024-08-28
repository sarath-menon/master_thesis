import base64
import io
from PIL import Image
from litellm import acompletion
import asyncio
from typing import Dict, Tuple
import json
from .caching import cache_result

class ImageProcessorBase:
    def __init__(self, model: str = "gpt-4o", temperature: float = 0.0):
        self.model = model
        self.temperature = temperature
        self.lock = asyncio.Lock()

    def _pil_to_base64(self, image: Image.Image) -> str:
        with io.BytesIO() as buffer:
            image.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode("utf-8")

    @cache_result(expiration_time=300)
    async def _get_image_response(self, base64_image: str, text_prompt: str, messages: list, json_mode: bool = False):
        msg = {
            "role": "user", 
            "content": [
                {"type": "text", "text": text_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
            ]
        }

        async with self.lock: 
            messages_copy = messages.copy()
            messages_copy.append(msg)

        response_format = {"type": "json_object"} if json_mode else None
        response = await acompletion(
            model=self.model, 
            messages=messages_copy, 
            temperature=self.temperature, 
            response_format=response_format
        )
        result = response["choices"][0]["message"]["content"]

        return result

    def clear_cache(self):
        self._get_image_response.clear_cache()