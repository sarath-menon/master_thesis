from clicking.common.bbox import BoundingBox, BBoxMode
import pytest

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

if __name__ == '__main__':
    pytest.main()
