from PIL import Image
import numpy as np
import os

from clicking.output_corrector.core import OutputCorrector
from clicking.pipeline.core import PipelineState
from clicking.common.data_structures import ClickingImage,ImageObject


def test_verify_bboxes_for_all_images():
    IMAGES_FOLDER_PATH = "./test/assets/crops"
    output_corrector = OutputCorrector(prompt_path="./prompts/output_corrector.md")
    
    # Load images and create ClickingImage objects
    clicking_images = []
    for filename in os.listdir(IMAGES_FOLDER_PATH):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(IMAGES_FOLDER_PATH, filename)
            image = Image.open(image_path)
            object_name = os.path.splitext(filename)[0]

        
            clicking_image = ClickingImage(
                id=object_name,
                image=image,
                predicted_objects=[
                    ImageObject(
                        name=object_name,
                    )
                ]
            )
            clicking_images.append(clicking_image)
    
    # Create a PipelineState object
    state = PipelineState(images=clicking_images)
    
    # Verify bboxes for all images
    results = []
    for clicking_image in state.images:
        verified_image = output_corrector.verify_bboxes(clicking_image)
        for obj in verified_image.predicted_objects:
            result = {
                'image_id': clicking_image.id,
                'object_name': obj.name,
                'judgement': obj.validity.is_valid,
                'reasoning': obj.validity.reason
            }
            results.append(result)
    
    # Print results
    for result in results:
        print(f"Image: {result['image_id']}")
        print(f"Object: {result['object_name']}")
        print(f"Valid: {result['judgement']}")
        print(f"Reason: {result['reasoning']}")
        print()
    
    # Assertions
    assert len(results) == len(clicking_images), "Number of results should match number of images"
    for result in results:
        assert 'image_id' in result
        assert 'object_name' in result
        assert 'judgement' in result
        assert 'reasoning' in result

# Run the test
test_verify_bboxes_for_all_images()