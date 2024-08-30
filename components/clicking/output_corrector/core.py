#%%
from litellm import completion, acompletion
import os
import dotenv
import base64
import io
from PIL import Image
import numpy as np
from clicking.vision_model.mask import SegmentationMask, SegmentationMode
from components.clicking.prompt_manager.core import PromptManager
import asyncio
from components.clicking.common.image_utils import ImageProcessorBase
from components.clicking.vision_model.visualization import overlay_bounding_box
from components.clicking.vision_model.core import LocalizationResults, SegmentationResults
from components.clicking.prompt_refinement.types import *

# set API keys
dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

class OutputCorrector(ImageProcessorBase):
    def __init__(self, prompt_path: str, model: str = "gpt-4o", temperature: float = 0.0):
        super().__init__(model, temperature)
        self.PROMPT_PATH = prompt_path
        self.prompt_manager = PromptManager(self.PROMPT_PATH)
        self.messages = [
            {"role": "system", "content": self.prompt_manager.get_prompt(type='system')},
        ]

    async def _get_image_responses(self, base64_images: list, object_names: list):
        tasks = []

        for base64_image, object_name in zip(base64_images, object_names):
            template_values = {"object_name": object_name}
            prompt = self.prompt_manager.get_prompt(type='user', prompt_key='correct_object_name', template_values=template_values)
            task = self._get_image_response(base64_image, prompt, self.messages, json_mode=True)
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks)
        return responses

    async def verify_bboxes_async(self, localization_results: LocalizationResults) -> LocalizationResults:
        verification_results = {}
        for sample in localization_results.processed_samples:
            image_id = sample.id
            screenshot = sample.image
            bboxes = localization_results.predictions[image_id]
            descriptions = [bbox.description for bbox in bboxes]
            
            images_overlayed = [overlay_bounding_box(screenshot.copy(), bbox) for bbox in bboxes]
            base64_images = [self._pil_to_base64(img) for img in images_overlayed]
            
            responses = await self._get_image_responses(base64_images, descriptions)
            verification_results[image_id] = responses
        
        return localization_results

    def verify_bboxes(self, localization_results: LocalizationResults) -> LocalizationResults:
        return asyncio.run(self.verify_bboxes_async(localization_results))

    async def verify_masks_async(self, localization_results: SegmentationResults) -> SegmentationResults:
        verification_results = {}
        for sample in localization_results.processed_samples:
            image_id = sample.id
            screenshot = sample.image
            masks = localization_results.predictions[image_id]
            object_names = [mask.object_name for mask in masks]
            
            extracted_areas = [mask.extract_area(screenshot, padding=10) for mask in masks]
            base64_images = [self._pil_to_base64(area) for area in extracted_areas]
            
            responses = await self._get_image_responses(base64_images, object_names)
            verification_results[image_id] = responses
        
        return verification_results

    def verify_masks(self, localization_results: SegmentationResults) -> SegmentationResults:
        return asyncio.run(self.verify_masks_async(localization_results))

#%%%

# Demo code for verify_bboxes_async
async def demo_verify_bboxes_async():
    # Create a sample LocalizationResults object
    from components.clicking.vision_model.core import ImageWithDescriptions, BoundingBox, BBoxMode
    from PIL import Image
    import numpy as np

    # Create a sample image
    sample_image = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
    
    # Create sample bounding boxes
    bbox1 = BoundingBox([10, 10, 50, 50], mode=BBoxMode.XYWH, description="Button")
    bbox2 = BoundingBox([50, 50, 40, 20], mode=BBoxMode.XYWH, description="Text Input")
    
    # Create a sample ImageWithDescriptions
    processed_sample = ImageWithDescriptions(
        id="sample1",
        image=sample_image,
        object_name="Sample Object",
        description="A sample processed image"
    )
    
    # Create a sample LocalizationResults object
    localization_results = LocalizationResults(
        processed_samples=[processed_sample],
        predictions={"sample1": [bbox1, bbox2]}
    )
    
    # Initialize OutputCorrector
    corrector = OutputCorrector(prompt_path="./prompts/output_corrector.md")
    
    # Call verify_bboxes_async
    verified_results = await corrector.verify_bboxes_async(localization_results)
    
    # Print the verification results
    for image_id, responses in verified_results.predictions.items():
        print(f"Verification results for image {image_id}:")
        for response in responses:
            print(response)

# Run the demo
if __name__ == "__main__":
    from nest_asyncio import apply
    apply()
    
    asyncio.run(demo_verify_bboxes_async())


# %%
