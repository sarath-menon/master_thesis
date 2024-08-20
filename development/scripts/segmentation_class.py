#%%

from enum import Enum
from typing import Union, List
import numpy as np
from pycocotools import mask as mask_utils
from PIL import Image
import matplotlib.pyplot as plt
from clicking.visualization.mask import SegmentationMask, SegmentationMode
#%%  create a sample mask

image = Image.open("./datasets/resized_media/gameplay_images/mario_odessey/8.jpg")
image_width, image_height = image.size

# Create a mask of the same size as the image
mask_array = np.zeros((image_height, image_width), dtype=bool)

# Define the size of the rectangle
rect_width, rect_height = 150, 250  # You can adjust the size as needed

# Generate a random position for the rectangle
x_start, y_start = 400, 300

# Set the rectangle area to True
mask_array[y_start:y_start+rect_height, x_start:x_start+rect_width] = True

# Create the SegmentationMask object
mask = SegmentationMask(mask_array, mode=SegmentationMode.BINARY_MASK)

extracted_image = mask.extract_area(image, padding=10)

fig, axs = plt.subplots(1, 3, figsize=(12, 6))

# Display the original image
axs[0].imshow(image)
axs[0].set_title('Original Image')
axs[0].axis('off')

# Display the mask
axs[1].imshow(mask.get(SegmentationMode.BINARY_MASK), cmap='gray')
axs[1].set_title('Mask')
axs[1].axis('off')

# Display the extracted image
axs[2].imshow(extracted_image)
axs[2].set_title('Extracted Area')
axs[2].axis('off')

plt.tight_layout()
plt.show()

# %%
