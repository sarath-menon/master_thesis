#%%
from litellm import acompletion, completion
import asyncio
import os
import dotenv
import base64
import io
import json
from clicking.visualization.mask import SegmentationMask, SegmentationMode
from clicking.output_corrector.core import OutputCorrector
# set API keys
dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


#%%
# from clicking.visualization.core import InstructionToLabel
from PIL import ImageDraw, Image
import matplotlib.pyplot as plt
from clicking.visualization.core import overlay_bounding_box
from clicking.visualization.bbox import BoundingBox, BBoxMode

bbox = BoundingBox((1000, 400, 200, 200), BBoxMode.XYWH)

image = Image.open("./datasets/resized_media/gameplay_images/mario_odessey/8.jpg")
image_overlayed = overlay_bounding_box(image.copy(), bbox)
plt.grid(False)
plt.axis('off')
plt.imshow(image_overlayed)

#%% verify bbox
output_corrector = OutputCorrector()
response = output_corrector.verify_bbox(image, "building")
print(response)
#%% Create a sample mask
import numpy as np

output_corrector = OutputCorrector()

mask_array = np.zeros((image.height, image.width), dtype=bool)
rect_width, rect_height = 120, 100  # You can adjust the size as needed
x_start, y_start = 1020, 440
mask_array[y_start:y_start+rect_height, x_start:x_start+rect_width] = True

# Create the SegmentationMask object
mask = SegmentationMask(mask_array, mode=SegmentationMode.BINARY_MASK)

extracted_image = mask.extract_area(image, padding=0)

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
#%% Verify mask

response = output_corrector.verify_mask(image, mask, "flag")
print(response)
# %%