#%%

from enum import Enum
from typing import Union, List
import numpy as np
from pycocotools import mask as mask_utils
from PIL import Image
import matplotlib.pyplot as plt
#%%
class SegmentationMode(Enum):
    BINARY_MASK = 1
    COCO_RLE = 2

class SegmentationMask:
    def __init__(self, mask: Union[np.ndarray, dict], mode: SegmentationMode = SegmentationMode.BINARY_MASK):
        if mode == SegmentationMode.BINARY_MASK:
            self._binary_mask = mask.astype(bool)
            self._coco_rle = mask_utils.encode(np.asfortranarray(self._binary_mask.astype(np.uint8)))
        elif mode == SegmentationMode.COCO_RLE:
            self._coco_rle = mask
            self._binary_mask = mask_utils.decode(self._coco_rle).astype(bool)
        else:
            raise ValueError("Invalid segmentation mask mode")

    def get(self, mode: SegmentationMode) -> Union[np.ndarray, dict]:
        if mode == SegmentationMode.BINARY_MASK:
            return self._binary_mask
        elif mode == SegmentationMode.COCO_RLE:
            return self._coco_rle
        else:
            raise ValueError("Invalid segmentation mask mode")

    def convert(self, mode: SegmentationMode) -> Union[np.ndarray, dict]:
        return self.get(mode)

    
    def crop_using_bbox(self, image_np, bbox, padding=10):
        x1, y1, x2, y2 = map(int, bbox)

        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(image_np.shape[1], x2 + padding)
        y2 = min(image_np.shape[0], y2 + padding)

        cropped_image = image_np[y1:y2, x1:x2]
        return cropped_image

    def extract_area(self, image: Image.Image, padding: int = 0) -> Image.Image:
        # Sanity checks
        if image.size != self._binary_mask.shape[::-1]:
            raise ValueError(f"Image size {image.size} and mask size {self._binary_mask.shape[::-1]} do not match")

        # Convert image to numpy array only if necessary
        image_array = np.array(image.convert("RGB") if image.mode != "RGB" else image)

        # Find bounding box of the mask
        rows, cols = np.nonzero(self._binary_mask)
        if len(rows) == 0 or len(cols) == 0:
            return Image.new('RGB', (1, 1), color='black')  # Return a 1x1 black image if mask is empty

        rmin, rmax, cmin, cmax = rows.min(), rows.max(), cols.min(), cols.max()

        cropped_image = self.crop_using_bbox(image_array, bbox=[cmin, rmin, cmax, rmax], padding=padding)

        return Image.fromarray(cropped_image)

    @property
    def shape(self) -> tuple:
        return self._binary_mask.shape

    def area(self) -> float:
        return mask_utils.area(self._coco_rle).item()

    def bbox(self) -> List[int]:
        return mask_utils.toBbox(self._coco_rle).tolist()

    def __repr__(self):
        return f"SegmentationMask(shape={self.shape}, mode={SegmentationMode.BINARY_MASK})"