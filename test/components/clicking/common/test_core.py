from clicking.common.bbox import BoundingBox, BBoxMode
from clicking.common.mask import SegmentationMask, SegmentationMode
import pytest
import numpy as np

@pytest.fixture
def bbox_xyxy():
    return BoundingBox(bbox=(10, 20, 30, 40), mode=BBoxMode.XYXY)

@pytest.fixture
def bbox_xywh():
    return BoundingBox(bbox=(10, 20, 20, 20), mode=BBoxMode.XYWH)

@pytest.fixture
def bbox_center():
    return BoundingBox(bbox=(20, 30, 20, 20), mode=BBoxMode.CENTER)

@pytest.fixture
def bbox_polygon():
    return BoundingBox(bbox=[(10, 20), (30, 20), (30, 40), (10, 40)], mode=BBoxMode.POLYGON)

@pytest.fixture
def binary_mask():
    return np.array([
        [0, 0, 0, 0],
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 0]
    ], dtype=bool)

@pytest.fixture
def segmentation_mask(binary_mask):
    return SegmentationMask(binary_mask=binary_mask)

def test_xyxy_mode(bbox_xyxy):
    assert bbox_xyxy.get(BBoxMode.XYXY) == (10, 20, 30, 40)

def test_xywh_mode(bbox_xywh):
    assert bbox_xywh.get(BBoxMode.XYWH) == (10, 20, 20, 20)
    assert bbox_xywh.get(BBoxMode.XYXY) == (10, 20, 30, 40)

def test_center_mode(bbox_center):
    assert bbox_center.get(BBoxMode.CENTER) == (20, 30, 20, 20)
    assert bbox_center.get(BBoxMode.XYXY) == (10, 20, 30, 40)

def test_polygon_mode(bbox_polygon):
    expected_polygon = [(10, 20), (30, 20), (30, 40), (10, 40)]
    assert bbox_polygon.get(BBoxMode.POLYGON) == expected_polygon
    assert bbox_polygon.get(BBoxMode.XYXY) == (10, 20, 30, 40)

def test_invalid_mode():
    with pytest.raises(ValueError):
        BoundingBox(bbox=(10, 20, 30, 40), mode="INVALID_MODE")

def test_repr(bbox_xyxy):
    expected_repr = "BoundingBox(xyxy=(10, 20, 30, 40), object_name=None, description=None)"
    assert repr(bbox_xyxy) == expected_repr

def test_segmentation_mask_creation(segmentation_mask):
    assert segmentation_mask.mode == SegmentationMode.BINARY_MASK
    assert segmentation_mask.shape == (4, 4)

def test_segmentation_mask_get(segmentation_mask):
    assert np.array_equal(segmentation_mask.get(SegmentationMode.BINARY_MASK), segmentation_mask.binary_mask)
    assert isinstance(segmentation_mask.get(SegmentationMode.COCO_RLE), dict)

def test_segmentation_mask_area(segmentation_mask):
    assert segmentation_mask.area() == 4

def test_segmentation_mask_bbox(segmentation_mask):
    assert segmentation_mask.bbox() == [1, 1, 2, 2]

def test_segmentation_mask_invalid_creation():
    with pytest.raises(ValueError):
        SegmentationMask()

if __name__ == '__main__':
    pytest.main()
