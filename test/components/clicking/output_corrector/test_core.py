from clicking.output_corrector import core
from clicking.vision_model.types import LocalizationResults, SegmentationResults
from clicking.common.bbox import BoundingBox, BBoxMode
from clicking.common.mask import SegmentationMask, SegmentationMode
from clicking.prompt_refinement.types import ImageWithDescriptions, SinglePromptResponse, ObjectDescription
from PIL import Image
import numpy as np

def test_verify_bboxes():
    # Create a mock OutputCorrector
    output_corrector = core.OutputCorrector(prompt_path="./prompts/output_corrector.md")
    
    # Create mock data
    processed_samples = [
        ImageWithDescriptions(
            image=Image.open("./assets/bus.jpg"),
            id="test_image_1",
            description=SinglePromptResponse(objects=[
                ObjectDescription(name="bus", category="vehicle", description="A bus")
            ]),
            object_name="bus"
        )
    ]
    
    predictions = {
        "test_image_1": [
            BoundingBox([10, 10, 50, 50], mode=BBoxMode.XYWH, object_name="bus", description="A bus")
        ]
    }
    
    localization_results = LocalizationResults(
        processed_samples=processed_samples,
        predictions=predictions
    )
    
    # Call the method
    result = output_corrector.verify_bboxes(localization_results)
    
    # Assert the result
    assert isinstance(result, LocalizationResults)
    assert len(result.predictions["test_image_1"]) == 1
    assert result.predictions["test_image_1"][0].object_name == "bus"