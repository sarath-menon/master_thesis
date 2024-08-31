import pytest
from clicking.dataset_creator import core
from clicking.common.types import ClickingImage, ImageObject
from clicking.common.mask import SegmentationMask
from clicking.common.bbox import BoundingBox

@pytest.fixture
def coco_dataset():
    data_dir = "./datasets/label_studio_gen/coco_dataset/images"
    annFile = "./datasets/label_studio_gen/coco_dataset/result.json"
    return core.CocoDataset(data_dir, annFile)

def test_coco_dataset_initialization(coco_dataset):
    assert isinstance(coco_dataset, core.CocoDataset)
    assert len(coco_dataset.all_object_names) > 0

def test_sample_dataset(coco_dataset):
    image_ids = [0, 1]
    samples = coco_dataset.sample_dataset(image_ids)
    
    assert len(samples) == 2
    assert all(isinstance(sample, ClickingImage) for sample in samples)
    assert all(sample.id in ['0', '1'] for sample in samples)

def test_create_image_objects(coco_dataset):
    # Mock annotation
    annotation = {
        'category_id': 1,
        'bbox': [10, 20, 30, 40],
        'segmentation': [[10, 10, 20, 20, 30, 30]],
        'image_id': 0
    }
    
    objects = coco_dataset._create_image_objects([annotation])
    
    assert len(objects) == 1
    assert isinstance(objects[0], ImageObject)
    assert isinstance(objects[0].bbox, BoundingBox)
    assert isinstance(objects[0].mask, SegmentationMask)

def test_get_ground_truth(coco_dataset):
    # Try multiple image_ids in case some don't have annotations
    for image_id in range(5):
        ground_truth = coco_dataset.get_ground_truth(image_id)
        
        assert isinstance(ground_truth, list)
        if ground_truth:  # Only check if the list is not empty
            assert all(isinstance(obj, ImageObject) for obj in ground_truth)
            for obj in ground_truth:
                assert isinstance(obj.mask, SegmentationMask)
                assert isinstance(obj.bbox, BoundingBox)
                assert isinstance(obj.name, str)
                assert isinstance(obj.category, core.ObjectCategory)
        
        if ground_truth:
            break  # Exit the loop if we found an image with annotations
    else:
        pytest.skip("No images with annotations found in the first 5 images")

# Keep the original test
def test_sample():
    assert core is not None

# Test descriptions:

# • test_coco_dataset_initialization: Verifies that the CocoDataset is correctly initialized with object names.
# • test_sample_dataset: Ensures that the sample_dataset method returns the correct number and type of samples.
# • test_create_image_objects: Checks if the _create_image_objects method correctly creates ImageObject instances from annotations.
# • test_get_ground_truth: Validates that the get_ground_truth method returns a list of ImageObjects with correct attributes.
# • test_sample: A basic test to ensure the core module is imported correctly.
