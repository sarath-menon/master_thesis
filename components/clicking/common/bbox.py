from enum import Enum
from typing import Tuple, List, Union, Optional
from dataclasses import dataclass
from PIL import Image
import json

class BBoxMode(Enum):
    XYWH = 1
    XYXY = 2
    MINMAX = 3
    CENTER = 4
    POLYGON = 5

@dataclass
class BoundingBox:
    bbox: Union[Tuple[float, float, float, float], List[Tuple[float, float]]]
    mode: BBoxMode = BBoxMode.XYXY
    object_name: Optional[str] = None
    description: Optional[str] = None

    def __post_init__(self):
        self.bbox = self._validate_bbox(self.bbox, self.mode)

    def _validate_bbox(self, bbox, mode):
        if mode == BBoxMode.XYWH:
            x1, y1, w, h = bbox
            return (x1, y1, x1 + w, y1 + h)
        elif mode == BBoxMode.XYXY or mode == BBoxMode.MINMAX:
            return bbox
        elif mode == BBoxMode.CENTER:
            cx, cy, w, h = bbox
            return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
        elif mode == BBoxMode.POLYGON:
            x1 = min(p[0] for p in bbox)
            y1 = min(p[1] for p in bbox)
            x2 = max(p[0] for p in bbox)
            y2 = max(p[1] for p in bbox)
            return (x1, y1, x2, y2)
        else:
            raise ValueError("Invalid bounding box mode")

    def get(self, mode: BBoxMode) -> Union[Tuple[float, float, float, float], List[Tuple[float, float]]]:
        x1, y1, x2, y2 = self.bbox
        if mode == BBoxMode.XYWH:
            return (x1, y1, x2 - x1, y2 - y1)
        elif mode == BBoxMode.XYXY or mode == BBoxMode.MINMAX:
            return (x1, y1, x2, y2)
        elif mode == BBoxMode.CENTER:
            w, h = x2 - x1, y2 - y1
            return ((x1 + x2) / 2, (y1 + y2) / 2, w, h)
        elif mode == BBoxMode.POLYGON:
            return [
                (x1, y1),
                (x2, y1),
                (x2, y2),
                (x1, y2)
            ]
        else:
            raise ValueError("Invalid bounding box mode")

    def __repr__(self):
        return f"BoundingBox(xyxy={self.get(BBoxMode.XYXY)}, object_name={self.object_name}, description={self.description})"

    def extract_area(self, image: Image.Image, padding: int = 0) -> Image.Image:
        x1, y1, x2, y2 = self.get(BBoxMode.XYXY)
        
        width, height = image.size
        
        x1 = max(0, int(x1) - padding)
        y1 = max(0, int(y1) - padding)
        x2 = min(width, int(x2) + padding)
        y2 = min(height, int(y2) + padding)
        
        return image.crop((x1, y1, x2, y2))

    def to_dict(self):
        return {
            'bbox': self.bbox,
            'mode': self.mode.name,
            'object_name': self.object_name,
            'description': self.description
        }

    def to_json(self):
        return json.dumps(self.to_dict())
