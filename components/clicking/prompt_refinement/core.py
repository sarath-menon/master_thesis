#%%
from components.clicking.common.image_utils import ImageProcessorBase
from litellm import completion, acompletion
import os
import dotenv
import base64
import io
from enum import Enum, auto
from clicking.prompt_manager.core import PromptManager
import asyncio
import nest_asyncio
from typing import List, Dict, Optional, TypedDict, Union, NamedTuple
import json
from clicking.dataset_creator.types import DatasetSample
from PIL import Image
import uuid

# set API keys
dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

class PromptMode(Enum):
    OBJECTS_LIST_TO_DESCRIPTIONS = "OBJECTS_LIST_TO_DESCRIPTIONS"
    IMAGE_TO_OBJECT_DESCRIPTIONS = "IMAGE_TO_OBJECT_DESCRIPTIONS"
    IMAGE_TO_OBJECTS_LIST = "IMAGE_TO_OBJECTS_LIST"

class ObjectDescription(TypedDict):
    name: str
    category: str
    description: str

class PromptResponse(TypedDict):
    objects: List[ObjectDescription]

class ProcessedSample(NamedTuple):
    image: Image.Image
    image_id: str
    class_label: str
    description: PromptResponse

class ProcessedResult(NamedTuple):
    samples: List[ProcessedSample]

class TemplateValues(TypedDict, total=False):
    input_description: str
    word_limit: str
    description_length: int

class PromptRefiner(ImageProcessorBase):
    def __init__(self, prompt_path: str, model: str = "gpt-4o", temperature: float = 0.0):
        super().__init__(model, temperature)
        self.prompt_manager = PromptManager(prompt_path)
        self.messages = [{"role": "system", "content": self.prompt_manager.get_prompt(type='system')}]
    
    async def process_prompts_async(self, dataset_sample: DatasetSample, mode: PromptMode = PromptMode.IMAGE_TO_OBJECT_DESCRIPTIONS, **kwargs) -> ProcessedResult:
        tasks = [self._process_single_prompt(image, mode, class_label, **kwargs) 
                 for image, class_label in zip(dataset_sample.images, dataset_sample.class_labels)]
        results = await asyncio.gather(*tasks)
        
        processed_samples = [
            ProcessedSample(
                image=image,
                image_id=str(uuid.uuid4()),  # Generate a unique ID for each image
                class_label=class_label,
                description=description
            )
            for image, class_label, description in zip(dataset_sample.images, dataset_sample.class_labels, results)
        ]
        
        return ProcessedResult(samples=processed_samples)

    async def _process_single_prompt(self, image: Image.Image, mode: PromptMode, class_label: Optional[str] = None, **kwargs) -> PromptResponse:
        base64_image = self._pil_to_base64(image)
        template_values = self._get_template_values(mode, class_label, **kwargs)
        prompt = self.prompt_manager.get_prompt(type='user', prompt_key=mode.value, template_values=template_values)
        response = await super()._get_image_response(base64_image, prompt, self.messages, json_mode=True)

        response_dict = json.loads(response)

        if mode == PromptMode.IMAGE_TO_OBJECT_DESCRIPTIONS:
            # sort objects by category
            response_dict['objects'] = sorted(response_dict['objects'], key=lambda x: x['category'])

        return response_dict

    def _get_template_values(self, mode: PromptMode, class_label: Optional[str], **kwargs) -> TemplateValues:
        if class_label is None:
            return {}
        elif mode == PromptMode.OBJECTS_LIST_TO_DESCRIPTIONS:
            return {"input_description": class_label, "word_limit": str(kwargs.get('word_limit', 10))}
        elif mode == PromptMode.IMAGE_TO_OBJECT_DESCRIPTIONS:
            return {"description_length": kwargs.get('description_length', 20)}
        raise ValueError(f"Invalid mode: {mode}")

    def show_messages(self) -> None:
        for message in self.messages:
            print(message)

    def process_prompts(self, dataset_sample: DatasetSample, mode: PromptMode = PromptMode.IMAGE_TO_OBJECT_DESCRIPTIONS, **kwargs) -> ProcessedResult:
        return asyncio.run(self.process_prompts_async(dataset_sample, mode, **kwargs))

# %% get expanded description from class label

if __name__ == "__main__":
    from PIL import Image
    import asyncio
    import nest_asyncio
    nest_asyncio.apply()

    image = Image.open("./datasets/resized_media/gameplay_images/mario_odessey/8.jpg")

    # Create an instance of PromptRefiner
    prompt_refiner = PromptRefiner(prompt_path="./prompts/prompt_refinement.md")

    # Define the batch of screenshots and corresponding input texts
    screenshots = [image, image]
    input_texts = [
        "car",
        "flag"
    ]

    # Define the mode and word limit for the prompts
    mode = PromptMode.OBJECTS_LIST_TO_DESCRIPTIONS
    word_limit = 5

    # Call process_prompts asynchronously
    async def process_batch_prompts():
        results = await prompt_refiner.process_prompts_async(screenshots, mode, input_texts=input_texts, word_limit=word_limit)
        print(results)

    # Run the asynchronous function
    asyncio.get_event_loop().run_until_complete(process_batch_prompts())

# %% get class labels from image

if __name__ == "__main__":
    from PIL import Image
    import asyncio
    import nest_asyncio
    nest_asyncio.apply()

    image_1 = Image.open("./datasets/resized_media/gameplay_images/unpacking/1.jpg")
    image_2 = Image.open("./datasets/resized_media/gameplay_images/mario_odessey/8.jpg")

    # Create an instance of PromptRefiner
    prompt_refiner = PromptRefiner(prompt_path="./prompts/prompt_refinement.md")

    # Define the batch of screenshots 
    images = [image_1, image_2]

    # Define the mode and word limit for the prompts
    mode = PromptMode.IMAGE_TO_OBJECT_DESCRIPTIONS

      # Call process_prompts asynchronously
    async def process_batch_prompts():
        results = await prompt_refiner.process_prompts_async(images, mode)
        for result in results:
            print(result)

    # Run the asynchronous function
    asyncio.get_event_loop().run_until_complete(process_batch_prompts())

