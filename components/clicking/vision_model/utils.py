import numpy as np
from typing import Dict, Any
from pycocotools import mask as mask_utils
import numpy as np
from scipy.ndimage import center_of_mass
from scipy.ndimage import distance_transform_edt
import io
import base64
from PIL import Image

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

def pil_to_base64(img, format="PNG"):
    buffered = io.BytesIO()
    img.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def base64_to_pil(base64_image):
    image_data = base64.b64decode(base64_image)
    image = Image.open(io.BytesIO(image_data))
    return image


def crop_image(image, start_x=0, end_x=None, start_y=0, crop_height=None, target_width=None):
    """
    Process screenshot with custom cropping and optional scaling
    
    Args:
        image_path: Path to the image file
        start_x: Left crop position
        end_x: Right crop position (if None, uses full width minus start_x)
        start_y: Starting y coordinate for crop
        crop_height: Height of the crop area
        target_width: Desired final width in pixels (maintains aspect ratio if specified)
    """
    img = image.convert('RGB')
    
    # Handle right side cropping
    if end_x is None:
        end_x = img.width - start_x
    
    # Use full height if not specified
    if crop_height is None:
        crop_height = img.height - start_y
    
    # Perform the crop
    img = img.crop((start_x, start_y, end_x, start_y + crop_height))
    
    # Scale if target width is specified
    if target_width:
        aspect_ratio = img.width / img.height
        target_height = int(target_width / aspect_ratio)
        img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)

    # print(f"Final image resolution: {img.size}")
    # print(f"Final aspect ratio: {img.height/img.width:.3f}")
    return img