import numpy as np
from typing import Dict, Any
from pycocotools import mask as mask_utils
import numpy as np
from scipy.ndimage import center_of_mass
from scipy.ndimage import distance_transform_edt
import io
import base64



def coco_encode_rle(mask: np.ndarray) -> Dict[str, Any]:
    binary_mask = mask.astype(bool)
    rle = mask_utils.encode(np.asfortranarray(binary_mask))
    rle['counts'] = rle['counts'].decode('utf-8')
    return rle


def get_mask_centroid(mask):
    # Compute the distance transform
    distances = distance_transform_edt(mask)
    
    # Find the maximum distance, and thus the center of the largest inscribed circle
    max_dist_idx = np.unravel_index(np.argmax(distances), distances.shape)
    circle_center = (max_dist_idx[1], max_dist_idx[0])  # (x, y) format

    return circle_center

def image_to_base64(img):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str