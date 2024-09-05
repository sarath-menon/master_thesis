from enum import Enum
from typing import Union, List, Optional
import numpy as np
from pycocotools import mask as mask_utils
from PIL import Image
from dataclasses import dataclass, field
import json

class SegmentationMode(Enum):
    BINARY_MASK = 1
    COCO_RLE = 2

@dataclass
class SegmentationMask:
    binary_mask: np.ndarray = field(default=None, repr=False)
    coco_rle: dict = field(default=None, repr=False)
    object_name: Optional[str] = None
    description: Optional[str] = None
    mode: SegmentationMode = SegmentationMode.BINARY_MASK

    def __post_init__(self):
        if self.binary_mask is None and self.coco_rle is None:
            raise ValueError("Either binary_mask or coco_rle must be provided")
        
        if isinstance(self.binary_mask, np.ndarray):
            self.binary_mask = self.binary_mask.astype(bool)
            self.coco_rle = mask_utils.encode(np.asfortranarray(self.binary_mask.astype(np.uint8)))
        elif isinstance(self.coco_rle, dict):
            self.binary_mask = mask_utils.decode(self.coco_rle).astype(bool)
        else:
            raise ValueError("Invalid mask type")

    def get(self, mode: SegmentationMode) -> Union[np.ndarray, dict]:
        if mode == SegmentationMode.BINARY_MASK:
            return self.binary_mask
        elif mode == SegmentationMode.COCO_RLE:
            return self.coco_rle
        else:
            raise ValueError("Invalid segmentation mask mode")

    def convert(self, mode: SegmentationMode) -> Union[np.ndarray, dict]:
        return self.get(mode)

    def crop_using_bbox(self, image_np, bbox, padding=0):
        x1, y1, x2, y2 = map(int, bbox)

        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(image_np.shape[1], x2 + padding)
        y2 = min(image_np.shape[0], y2 + padding)

        cropped_image = image_np[y1:y2, x1:x2]
        return cropped_image

    def extract_area(self, image: Image.Image, padding: int = 0) -> Image.Image:
        if image.size != self.binary_mask.shape[::-1]:
            raise ValueError(f"Image size {image.size} and mask size {self.binary_mask.shape[::-1]} do not match")

        image_array = np.array(image.convert("RGB") if image.mode != "RGB" else image)

        rows, cols = np.nonzero(self.binary_mask)
        if len(rows) == 0 or len(cols) == 0:
            return Image.new('RGB', (1, 1), color='black')

        rmin, rmax, cmin, cmax = rows.min(), rows.max(), cols.min(), cols.max()

        cropped_image = self.crop_using_bbox(image_array, bbox=[cmin, rmin, cmax, rmax], padding=padding)

        return Image.fromarray(cropped_image)

    @property
    def shape(self) -> tuple:
        return self.binary_mask.shape

    def area(self) -> float:
        return mask_utils.area(self.coco_rle).item()

    def bbox(self) -> List[int]:
        return mask_utils.toBbox(self.coco_rle).tolist()

    def __repr__(self):
        return f"SegmentationMask(shape={self.shape}, mode={self.mode})"

    def to_dict(self):
        return {
            'mask': self.coco_rle,
            'mode': 'coco_rle',
            'object_name': self.object_name,
            'description': self.description
        }

    def to_json(self):
        return json.dumps(self.to_dict())
