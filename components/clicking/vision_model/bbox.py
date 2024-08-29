from enum import Enum
from typing import Tuple, List, Union

class BBoxMode(Enum):
    XYWH = 1
    XYXY = 2
    MINMAX = 3
    CENTER = 4
    POLYGON = 5

class BoundingBox:
    def __init__(self, bbox: Union[Tuple[float, float, float, float], List[Tuple[float, float]]], mode: BBoxMode = BBoxMode.XYXY):
        if mode == BBoxMode.XYWH:
            self._x1, self._y1, w, h = bbox
            self._x2, self._y2 = self._x1 + w, self._y1 + h
        elif mode == BBoxMode.XYXY or mode == BBoxMode.MINMAX:
            self._x1, self._y1, self._x2, self._y2 = bbox
        elif mode == BBoxMode.CENTER:
            cx, cy, w, h = bbox
            self._x1 = cx - w / 2
            self._y1 = cy - h / 2
            self._x2 = cx + w / 2
            self._y2 = cy + h / 2
        elif mode == BBoxMode.POLYGON:
            self._x1 = min(p[0] for p in bbox)
            self._y1 = min(p[1] for p in bbox)
            self._x2 = max(p[0] for p in bbox)
            self._y2 = max(p[1] for p in bbox)
        else:
            raise ValueError("Invalid bounding box mode")

    def get(self, mode: BBoxMode) -> Union[Tuple[float, float, float, float], List[Tuple[float, float]]]:
        if mode == BBoxMode.XYWH:
            return (self._x1, self._y1, self._x2 - self._x1, self._y2 - self._y1)
        elif mode == BBoxMode.XYXY or mode == BBoxMode.MINMAX:
            return (self._x1, self._y1, self._x2, self._y2)
        elif mode == BBoxMode.CENTER:
            w, h = self._x2 - self._x1, self._y2 - self._y1
            return ((self._x1 + self._x2) / 2, (self._y1 + self._y2) / 2, w, h)
        elif mode == BBoxMode.POLYGON:
            return [
                (self._x1, self._y1),
                (self._x2, self._y1),
                (self._x2, self._y2),
                (self._x1, self._y2)
            ]
        else:
            raise ValueError("Invalid bounding box mode")