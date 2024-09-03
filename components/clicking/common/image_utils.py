import base64
import io
from PIL import Image
from litellm import acompletion
import asyncio
from typing import Dict, Tuple, Type, TypeVar, Any
import json
from .caching import cache_result

T = TypeVar('T')

class ImageProcessorBase:
    def __init__(self, model: str = "gpt-4o", temperature: float = 0.0):
        self.model = model
        self.temperature = temperature

    def _pil_to_base64(self, image: Image.Image) -> str:
        with io.BytesIO() as buffer:
            image.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode("utf-8")

    @cache_result(expiration_time=3000)
    async def _get_image_response(self, image: Image.Image, text_prompt: str, messages: list, output_type: Type[T]) -> T:
        
        base64_image = self._pil_to_base64(image)
        msg = {
            "role": "user", 
            "content": [
                {"type": "text", "text": text_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
            ]
        }

        messages.append(msg)

        response_format = {"type": "json_object"}
        response = await acompletion(
            model=self.model, 
            messages=messages, 
            temperature=self.temperature, 
            response_format=response_format
        )
        result = response["choices"][0]["message"]["content"]

        try:
            parsed_result = json.loads(result)
            return output_type(**parsed_result)
        except (json.JSONDecodeError, TypeError) as e:
            raise ValueError(f"Response does not match the specified output type: {e}")

    def clear_cache(self):
        self._get_image_response.clear_cache()
