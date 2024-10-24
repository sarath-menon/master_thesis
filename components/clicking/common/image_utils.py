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
from typing import Optional

T = TypeVar('T')

import os

class ImageProcessorBase:
    def __init__(self, model: str = "gpt-4o-2024-08-06", temperature: float = 0.0):
        self.model = model
        self.temperature = temperature

    def _pil_to_base64(self, image: Image.Image) -> str:
        with io.BytesIO() as buffer:
            image.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode("utf-8")

    # @cache_result(expiration_time=3000)
    async def _get_image_response(self, image: Image.Image, text_prompt: str, messages: list, output_type: Optional[Type[T]] = None) -> T:
        
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

        if output_type is None:
            return result

        try:
            parsed_result = json.loads(result)
            return output_type(**parsed_result)
        except (json.JSONDecodeError, TypeError) as e:
            raise ValueError(f"Response does not match the specified output type: {e}")

    def clear_cache(self):
        self._get_image_response.clear_cache()

    @cache_result(expiration_time=3000)
    async def _get_batch_image_responses(self, images: List[Image.Image], text_prompts: List[str], messages: List[List[Dict]], output_type: Optional[Type[T]] = None) -> List[T]:
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

            if output_type is None:
                results.append(result)
                continue

            try:
                parsed_result = json.loads(result)
                results.append(output_type(**parsed_result))
            except Exception as e:
                print(f"Error in parsing response: {e}")
                continue

        return results


def crop_image(image, start_x=0, end_x=None, start_y=0, crop_height=None, target_width=None):
    """
    Process screenshot with custom cropping and optional scaling
    
    Args:
        image_path: Path to the image file
        start_x: Left crop position
        end_x: Right crop position (if None, uses full width minus start_x)
        start_y: Starting y coordinate for crop
        crop_height: Height of the crop area
        target_width: Desired final width in pixels (maintains aspect ratio if specified)
    """
    img = image.convert('RGB')
    
    # Handle right side cropping
    if end_x is None:
        end_x = img.width - start_x
    
    # Use full height if not specified
    if crop_height is None:
        crop_height = img.height - start_y
    
    # Perform the crop
    img = img.crop((start_x, start_y, end_x, start_y + crop_height))
    
    # Scale if target width is specified
    if target_width:
        aspect_ratio = img.width / img.height
        target_height = int(target_width / aspect_ratio)
        img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)

    print(f"Final image resolution: {img.size}")
    print(f"Final aspect ratio: {img.height/img.width:.3f}")
    return img