#%%
from litellm import completion, acompletion
import os
import dotenv
import base64
import io
from PIL import Image
import numpy as np
from clicking.common.mask import SegmentationMask, SegmentationMode
from clicking.prompt_manager.core import PromptManager
import asyncio
from clicking.common.image_utils import ImageProcessorBase
from clicking.vision_model.visualization import overlay_bounding_box
from clicking.common.types import ClickingImage
from clicking.prompt_refinement.types import *
import json
from pydantic import BaseModel, Field
from typing import Literal
from clicking.common.types import Validity

# set API keys
dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

class CorrectedResponse(BaseModel):
    judgement: Literal["true", "false"] = "true"
    reasoning: str

class BBoxVerificationMode(str):
    OVERLAY = "bbox_overlay"
    CROP = "bbox_crop"

class OutputCorrector(ImageProcessorBase):
    def __init__(self, prompt_path: str, model: str = "gpt-4o", temperature: float = 0.0):
        super().__init__(model, temperature)
        self.PROMPT_PATH = prompt_path
        self.prompt_manager = PromptManager(self.PROMPT_PATH)
        self.system_message = {"role": "system", "content": self.prompt_manager.get_prompt(type='system')}

    async def verify_bboxes_async(self, clicking_image: ClickingImage, mode: BBoxVerificationMode = BBoxVerificationMode.CROP) -> ClickingImage:
        screenshot = clicking_image.image

        print(f"Predicted objects: {clicking_image.predicted_objects}")
        
        if mode == BBoxVerificationMode.OVERLAY:
            images = [overlay_bounding_box(screenshot.copy(), obj.bbox) for obj in clicking_image.predicted_objects]
        elif mode == BBoxVerificationMode.CROP:
            images = [obj.bbox.extract_area(screenshot.copy(), padding=0) for obj in clicking_image.predicted_objects]
        else:
            raise ValueError(f"Invalid mode: {mode}")
        
        import matplotlib.pyplot as plt

        async def process_object(image, obj):
            template_values = {"object_name": obj.name}
            prompt = self.prompt_manager.get_prompt(type='user', prompt_key=mode, template_values=template_values)

            # Create a new message list for each object
            messages = [self.system_message, {"role": "user", "content": prompt}]

            response: CorrectedResponse = await self._get_image_response(image, prompt, messages, output_type=CorrectedResponse)

            obj.validity.is_valid = response.judgement != "false"
            obj.validity.reason = response.reasoning

            plt.axis('off')
            plt.imshow(image)
            plt.title(f"Object {obj.name}")
            plt.show()

        tasks = [process_object(image, obj) for image, obj in zip(images, clicking_image.predicted_objects)]
        await asyncio.gather(*tasks)

        return clicking_image

    def verify_bboxes(self, clicking_image: ClickingImage) -> ClickingImage:
        return asyncio.run(self.verify_bboxes_async(clicking_image))

    async def verify_masks_async(self, clicking_image: ClickingImage) -> ClickingImage:
        verification_results = {}
        screenshot = clicking_image.image
        
        extracted_areas = [obj.mask.extract_area(screenshot, padding=10) for obj in clicking_image.predicted_objects]
        base64_images = [self._pil_to_base64(area) for area in extracted_areas]
        object_names = [obj.name for obj in clicking_image.predicted_objects]
        
        responses = await self._get_image_responses(base64_images, object_names)
        verification_results[clicking_image.id] = responses

        for response in responses:
            print(response)
        
        return clicking_image

    def verify_masks(self, clicking_image: ClickingImage) -> ClickingImage:
        return asyncio.run(self.verify_masks_async(clicking_image))

#%%%

# Demo code for verify_bboxes_async
async def demo_verify_bboxes_async():
    from clicking.common.types import ClickingImage, ImageObject, ObjectCategory
    from clicking.common.bbox import BoundingBox, BBoxMode
    from PIL import Image
    import numpy as np

    # Create a sample image
    sample_image = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
    
    # Create sample ImageObjects
    obj1 = ImageObject(
        name="Button",
        description="A button",
        category=ObjectCategory.GAME_ASSET,
        bbox=BoundingBox([10, 10, 50, 50], mode=BBoxMode.XYWH),
        mask=SegmentationMask(np.zeros((100, 100)), mode=SegmentationMode.BINARY_MASK)
    )
    obj2 = ImageObject(
        name="Text Input",
        description="A text input field",
        category=ObjectCategory.GAME_ASSET,
        bbox=BoundingBox([50, 50, 40, 20], mode=BBoxMode.XYWH),
        mask=SegmentationMask(np.zeros((100, 100)), mode=SegmentationMode.BINARY_MASK)
    )
    
    # Create a sample ClickingImage
    clicking_image = ClickingImage(
        id="sample1",
        image=sample_image,
        objects=[obj1, obj2]
    )
    
    # Initialize OutputCorrector
    corrector = OutputCorrector(prompt_path="./prompts/output_corrector.md")
    
    # Call verify_bboxes_async
    verified_image = await corrector.verify_bboxes_async(clicking_image)
    
    # Print the verification results
    print(f"Verification results for image {verified_image.id}:")
    for obj in verified_image.predicted_objects:
        print(f"Object: {obj.name}")
        print(f"BBox: {obj.bbox}")
        print("---")

# Run the demo
if __name__ == "__main__":
    from nest_asyncio import apply
    apply()
    
    asyncio.run(demo_verify_bboxes_async())

# %%



