import numpy as np
from scipy.ndimage import center_of_mass

# return the centroid of a mask as a (x, y)
def get_mask_centroid(mask):
    # Ensure the mask is boolean
    mask = mask.astype(bool)
    
    # Calculate the center of mass
    centroid = center_of_mass(mask)
    centroid = (centroid[2], centroid[1])

    # if mask is not in the centroid, find the nearest true pixel
    if not mask[centroid]:
        true_indices = np.argwhere(mask)
        centroid = true_indices[np.argmin(np.sum((true_indices - np.array(centroid))**2, axis=1))]

    return centroid