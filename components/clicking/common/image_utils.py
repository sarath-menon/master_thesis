import base64
import io
from PIL import Image
from litellm import acompletion
import asyncio
from typing import Dict, Tuple, Type, TypeVar, Any, List
import json
from .caching import cache_result
from litellm import batch_completion
import openai

T = TypeVar('T')

import os

class ImageProcessorBase:
    def __init__(self, model: str = "gpt-4o", temperature: float = 0.0):
        self.model = model
        self.temperature = temperature

    def _pil_to_base64(self, image: Image.Image) -> str:
        with io.BytesIO() as buffer:
            image.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode("utf-8")

    # @cache_result(expiration_time=3000)
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
            response_format=response_format,
            num_retries=3,
            timeout=60,
            caching=True,
        )
        result = response["choices"][0]["message"]["content"]

        try:
            parsed_result = json.loads(result)
            return output_type(**parsed_result)
        except (json.JSONDecodeError, TypeError) as e:
            raise ValueError(f"Response does not match the specified output type: {e}")

    def clear_cache(self):
        self._get_image_response.clear_cache()

    @cache_result(expiration_time=3000)
    async def _get_batch_image_responses(self, images: List[Image.Image], text_prompts: List[str], messages: List[List[Dict]], output_type: Type[T]) -> List[T]:
        base64_images = [self._pil_to_base64(img) for img in images]
        
        batch_messages = []
        for base64_image, text_prompt, msg_list in zip(base64_images, text_prompts, messages):
            msg = {
                "role": "user",
                "content": [
                    {"type": "text", "text": text_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                ]
            }
            batch_messages.append(msg_list + [msg])

        response_format = {"type": "json_object"}

        try:
            responses = batch_completion(
                model=self.model,
                messages=batch_messages,
                temperature=self.temperature,
                response_format=response_format,
                num_retries=3,
            )

        except openai.RateLimitError as e:
            print("Passed: Raised correct exception. Got openai.RateLimitError\nGood Job", e)
            print(type(e))
            return []
        except Exception as e:
            print(f"Error in batch processing: {type(e)}, Error: {e}")
            return []

        results = []
        for response in responses:
            if isinstance(response, Exception):
                print(f"Error in response: {response}")
                continue
            result = response["choices"][0]["message"]["content"]

            try:
                parsed_result = json.loads(result)
                results.append(output_type(**parsed_result))
            except Exception as e:
                print(f"Error in parsing response: {e}")
                continue

        return results
