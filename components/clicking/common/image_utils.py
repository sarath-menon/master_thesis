import base64
import io
from PIL import Image
from litellm import acompletion
import asyncio
import time
import hashlib
from typing import Dict, Tuple
import json

class ImageProcessorBase:
    def __init__(self, model: str = "gpt-4o", temperature: float = 0.0):
        self.model = model
        self.temperature = temperature
        self.lock = asyncio.Lock()
        self.prediction_cache: Dict[str, Tuple[str, float]] = {}
        self.CACHE_EXPIRATION_TIME = 300  # 5 minutes in seconds

    def _pil_to_base64(self, image: Image.Image) -> str:
        with io.BytesIO() as buffer:
            image.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode("utf-8")

    async def _get_image_response(self, base64_image: str, text_prompt: str, messages: list, json_mode: bool = False):
        cache_key = self._generate_cache_key(base64_image, text_prompt, json.dumps(messages), json_mode)

        # Check if the prediction is in the cache and not expired
        cached_result = self.prediction_cache.get(cache_key)
        if cached_result:
            prediction, timestamp = cached_result
            if time.time() - timestamp < self.CACHE_EXPIRATION_TIME:
                print("Using cached result")
                return prediction

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

        # Cache the new prediction
        self.prediction_cache[cache_key] = (result, time.time())
        print("Caching new result")

        return result

    def _generate_cache_key(self, *args) -> str:
        key = hashlib.md5()
        for arg in args:
            if isinstance(arg, list):
                for item in arg:
                    key.update(str(item).encode())
            else:
                key.update(str(arg).encode())
        return key.hexdigest()

    def clear_cache(self):
        self.prediction_cache.clear()