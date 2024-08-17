import numpy as np
from scipy.ndimage import center_of_mass

from scipy.ndimage import distance_transform_edt

def get_mask_centroid(mask):
    # Compute the distance transform
    distances = distance_transform_edt(mask)
    
    # Find the maximum distance, and thus the center of the largest inscribed circle
    max_dist_idx = np.unravel_index(np.argmax(distances), distances.shape)
    circle_center = (max_dist_idx[1], max_dist_idx[0])  # (x, y) format

    return circle_center