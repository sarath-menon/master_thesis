#%%
import os
from litellm import completion
import json
#%%

messages = [{"role": "user", "content": "Hey! how's it going?"}]
response = completion(model="claude-3-5-sonnet-20241022", messages=messages, system="You are a helpful assistant.")
print(response.choices[0].message.content)
# %%

from clicking.common.image_utils import ImageProcessorBase
from PIL import Image
client = ImageProcessorBase(model="claude-3-5-sonnet-20241022")

#%%

messages = [{"role": "system", "content": 
"""
You are an expert at locating objects in images. Your task is to identify the precise pixel coordinates of objects in images.

Input:
- An image
- The name of an object to locate

Output Requirements:
1. Provide the exact pixel coordinates (x,y) for the center point of the specified object
2. The coordinate system starts at (0,0) in the top-left corner of the image
3. X coordinates increase from left to right
4. Y coordinates increase from top to bottom
5. Coordinates must be integers representing exact pixel positions

Guidelines:
- Be as precise as possible - do not provide rough estimates
- Always aim for the center point of the object
- If multiple instances of the object exist, specify which one you are referring to
- If the object is not visible or cannot be found, explicitly state this
"""
}]

def get_text_prompt(object_name: str):
    return f"""
    The object to locate is a {object_name}. Analyze the image and provide the following JSON output:
    {{
        "reasoning": "Explanation of how you identified the object's location in 20 words",
        "x": "Exact pixel x-coordinate of object's center (integer)",
        "y": "Exact pixel y-coordinate of object's center (integer)"
    }}
    """

text_prompt = get_text_prompt("button named I AGREE")
# image = Image.open("./datasets/anthropic_clicking/anthropic_demo.jpg")
image = Image.open("./datasets/resized_media/monopoly_images/51.jpg")
centered_image, offset, scale_factor = create_centered_image(image)

response = await client.get_image_response(image=centered_image, messages=messages, text_prompt=text_prompt)

response = json.loads(response)

# convert to pixel coordinates in the original image
x = (response["x"] - offset[0]) * scale_factor
y = (response["y"] - offset[1]) * scale_factor
print(f"x: {x}, y: {y}")

plot_image_with_point(image, x, y)
# plot_image_with_point(centered_image, response["x"], response["y"])

#%%

def create_centered_image(input_image: Image.Image, canvas_width: int = 1024, canvas_height: int = 768) -> Image.Image:
    # Create transparent background
    background = Image.new('RGBA', (canvas_width, canvas_height), (0, 0, 0, 0))
    
    # Scale image if larger than canvas
    scale_factor = 4
    if input_image.width > canvas_width or input_image.height > canvas_height:
        input_image = input_image.resize((input_image.width // scale_factor, input_image.height // scale_factor))
    
    # Calculate position to paste input image
    x_offset = (canvas_width - input_image.width) // 2
    y_offset = (canvas_height - input_image.height) // 2
    
    # Create a copy of the background and paste input image
    result = background.copy()
    result.paste(input_image, (x_offset, y_offset))
    
    return result, [x_offset, y_offset], scale_factor


image = Image.open("./datasets/resized_media/monopoly_images/51.jpg")
image, offset = create_centered_image(image)
image
#%%
import matplotlib.pyplot as plt

def plot_image_with_point(image, x, y):
    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    plt.grid(False)
    plt.axis('off')

    plt.plot(x, y, marker='*', color='red', markersize=15)
    
    plt.show()


# %%
