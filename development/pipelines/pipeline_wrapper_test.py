#%%
%load_ext autoreload
%autoreload 2

import nest_asyncio
nest_asyncio.apply()
import yaml

from clicking.pipelines.molmo_direct import MolmoDirectPipelineWrapper
# Load the configuration file
CONFIG_PATH = "./development/pipelines/game_object_config.yml"
with open(CONFIG_PATH, 'r') as config_file:
    config = yaml.safe_load(config_file)

#%%
pipeline_wrapper = MolmoDirectPipelineWrapper(config)
#%%
import asyncio
from PIL import Image

img = Image.open("./datasets/resized_media/gameplay_images/mario_odessey/0.jpg")
text_input = "moon"
# Process the image using the pipeline wrapper
clickpoint = asyncio.run(pipeline_wrapper.process_image(img, text_input))

print(clickpoint)

#%%
import matplotlib.pyplot as plt
def plot_image_with_point(image, point):
    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    plt.grid(False)
    plt.axis('off')

    # Convert the point coordinates to image coordinates
    img_width, img_height = image.size
    x_img = point.x/100 * img_width
    y_img = point.y/100 * img_height
    
    plt.plot(x_img, y_img, marker='*', color='red', markersize=15)
    
    plt.show()

plot_image_with_point(img, clickpoint)

# %%
# compare aspect ratios of the image and the screenshot

def print_image_info(image):
    plt.imshow(image)
    plt.grid(False)
    plt.axis('off')
    plt.show()
    print(f"Image size: {image.size}")
    print(f"Aspect ratio: {image.size[0]/image.size[1]}")

# get a 
screenshot = Image.open("/Users/sarathmenon/Desktop/Screenshot 2024-09-29 at 9.27.36 PM.jpg")


print_image_info(screenshot)
# %%
from PIL import Image
from clicking.pipelines.molmo_direct import ClickPoint

def transform_clickpoint(original_image, clickpoint: ClickPoint, screenshot):
    """
    Transform a clickpoint from a downsampled image to its position in a higher resolution screenshot
    with potentially different aspect ratio and black borders.
    
    :param original_image: The original downsampled PIL Image
    :param clickpoint: ClickPoint representing the clickpoint in the original image
    :param screenshot: The target screenshot PIL Image
    :return: ClickPoint representing the transformed clickpoint
    """
    macbook_height = 2234
    ryujinx_height = 2160
    y_offset = (macbook_height - ryujinx_height) / 2

    original_width, original_height = original_image.size
    target_width, target_height = screenshot.size
    
    # Calculate aspect ratios
    original_ratio = original_width / original_height
    target_ratio = target_width / target_height
    
    # Calculate the dimensions of the game area in the screenshot (excluding black borders)
    if original_ratio > target_ratio:
        # Black borders on top and bottom
        screenshot_game_width = target_width
        screenshot_game_height = int(screenshot_game_width / original_ratio)
        border_height = (target_height - screenshot_game_height) // 2
        border_width = 0
    else:
        # Black borders on left and right
        screenshot_game_height = target_height
        screenshot_game_width = int(screenshot_game_height * original_ratio)
        border_width = (target_width - screenshot_game_width) // 2
        border_height = 0
    
    # Calculate the scaling factors
    scale_x = screenshot_game_width / original_width
    scale_y = screenshot_game_height / original_height
    
    # Transform clickpoint
    x = (clickpoint.x / 100) * original_width
    y = (clickpoint.y / 100) * original_height
    
    new_x = (x * scale_x) + border_width 
    new_y = (y * scale_y) + border_height + y_offset
    
    # Convert back to 0-100 scale
    new_x = (new_x / target_width) * 100
    new_y = (new_y / target_height) * 100
    
    # Ensure the new clickpoint is within the target image bounds
    new_x = max(0, min(new_x, 100))
    new_y = max(0, min(new_y, 100))
    
    return ClickPoint(name="", x=new_x, y=new_y)

# Example usage:
original_image = Image.open("/Users/sarathmenon/Downloads/image.webp")
screenshot = Image.open("/Users/sarathmenon/Desktop/123.jpg")
original_clickpoint = ClickPoint(name="", x=58, y=30)

new_clickpoint = transform_clickpoint(original_image, original_clickpoint, screenshot)
print(f"Original clickpoint: {original_clickpoint}")
print(f"Transformed clickpoint: {new_clickpoint}")

plot_image_with_point(original_image, original_clickpoint)
plot_image_with_point(screenshot, new_clickpoint)

# %%