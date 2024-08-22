import numpy as np
from typing import Dict, Any
from pycocotools import mask as mask_utils

def coco_encode_rle(mask: np.ndarray) -> Dict[str, Any]:
    binary_mask = mask.astype(bool)
    rle = mask_utils.encode(np.asfortranarray(binary_mask))
    rle['counts'] = rle['counts'].decode('utf-8')
    return rle