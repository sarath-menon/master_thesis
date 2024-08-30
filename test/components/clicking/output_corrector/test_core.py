from clicking.output_corrector import core
from clicking.vision_model.types import LocalizationResults, SegmentationResults
from clicking.vision_model.bbox import BoundingBox, BBoxMode
from clicking.vision_model.mask import SegmentationMask, SegmentationMode
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
            image_id="test_image_1",
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

# Uncomment and update this test when implementing verify_masks
# def test_verify_masks():
#     # Create a mock OutputCorrector
#     output_corrector = core.OutputCorrector(prompt_path="./prompts/output_corrector.md")
#     
#     # Create mock data
#     processed_samples = [
#         ImageWithDescriptions(
#             image=Image.new('RGB', (100, 100)),
#             image_id="test_image_1",
#             description=SinglePromptResponse(objects=[
#                 ObjectDescription(name="dog", category="animal", description="A dog running")
#             ]),
#             object_name="dog"
#         )
#     ]
#     
#     mock_mask = np.zeros((100, 100), dtype=np.uint8)
#     mock_mask[25:75, 25:75] = 1
#     
#     predictions = {
#         "test_image_1": [
#             SegmentationMask(mask=mock_mask, mode=SegmentationMode.BINARY_MASK, object_name="dog", description="A dog running")
#         ]
#     }
#     
#     segmentation_results = SegmentationResults(
#         processed_samples=processed_samples,
#         predictions=predictions
#     )
#     
#     # Call the method
#     result = output_corrector.verify_masks(segmentation_results)
#     
#     # Assert the result
#     assert isinstance(result, SegmentationResults)
#     assert len(result.predictions["test_image_1"]) == 1
#     assert result.predictions["test_image_1"][0].object_name == "dog"
