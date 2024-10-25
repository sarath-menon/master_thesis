#%%
import os
from litellm import completion
import json
#%%

# messages = [{"role": "user", "content": "Hey! how's it going?"}]
# response = completion(model="claude-3-5-sonnet-20241022", messages=messages, system="You are a helpful assistant.")
# print(response.choices[0].message.content)

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
            "object_id": "1",
            "reasoning": "Explanation of how you identified the object's location in 20 words",
            "x": "Exact pixel x-coordinate of object's center (integer)",
            "y": "Exact pixel y-coordinate of object's center (integer)"
        }}
    """

object_name = "silver hat at the top"
text_prompt = get_text_prompt(object_name)

text_prompt = text_prompt.strip()

image = Image.open("./datasets/resized_media/monopoly_images/14.jpg")
centered_image, offset, scale_factor = create_centered_image(image)

response = await client.get_image_response(image=centered_image, messages=messages, text_prompt=text_prompt)

response = json.loads(response)
print(f"x: {response['x']}, y: {response['y']}")

# convert to pixel coordinates in the original image
x = (response["x"] - offset[0]) * scale_factor
y = (response["y"] - offset[1]) * scale_factor
print(f"x: {x}, y: {y}")

plot_image(image, x, y)
print(object_name)
# plot_image_with_point(centered_image, response["x"], response["y"])
#%%
text_prompt
#%%
from clicking.prompt_manager.core import PromptManager
import yaml
# Load the configuration file«
CONFIG_PATH = "./development/pipelines/monopoly_config.yml"
with open(CONFIG_PATH, 'r') as config_file:
    config = yaml.safe_load(config_file)

prompt_manager = PromptManager(config['prompts']['anthropic_clicking_path'])
batch_messages = [{"role": "system", "content":  prompt_manager.get_prompt(type='system')}]

image1 = Image.open("./datasets/resized_media/monopoly_images/14.jpg")
centered_image1, offset1, scale_factor1 = create_centered_image(image1)

image2 = Image.open("./datasets/resized_media/monopoly_images/18.jpg")
centered_image2, offset2, scale_factor2 = create_centered_image(image2)

image3 = Image.open("./datasets/resized_media/monopoly_images/24.jpg")
centered_image3, offset3, scale_factor3 = create_centered_image(image3)

object_name1 = "silver hat at the top"
object_name2 = "green button named COLLECT"
object_name3 = "coin in the first row"

text_prompt1 = prompt_manager.get_prompt(type='user', prompt_key='default', template_values={"object_name": object_name1, "object_id": "1"})
text_prompt2 = prompt_manager.get_prompt(type='user', prompt_key='default', template_values={"object_name": object_name2, "object_id": "2"})
text_prompt3 = prompt_manager.get_prompt(type='user', prompt_key='default', template_values={"object_name": object_name3, "object_id": "3"})


images = [centered_image1, centered_image2, centered_image3]
text_prompts= [text_prompt1, text_prompt2, text_prompt3]
batch_messages = [messages.copy(), messages.copy(), messages.copy()]

# add newline to the beginning and end of the prompts if they don't already have them
for i, text_prompt in enumerate(text_prompts):
    if not text_prompt.startswith("\n"):
        text_prompts[i] = "\n" + text_prompt
    if not text_prompt.endswith("\n"):
        text_prompts[i] = text_prompt + "\n"


responses = await client._get_batch_image_responses(images=images, text_prompts=text_prompts, messages=batch_messages)

original_images = [image1, image2, image3]
offsets = [offset1, offset2, offset3]    
scale_factors = [scale_factor1, scale_factor2, scale_factor3]

for resp, image, offset, scale_factor in zip(responses, original_images, offsets, scale_factors):
    resp = json.loads(resp)
    x = (resp["x"] - offset[0]) * scale_factor
    y = (resp["y"] - offset[1]) * scale_factor
    print(f"{object_name}: {x}, {y}")
    plot_image(image, x, y)

#%%



# print(text_prompt)
# print(text_prompt1)

def show_text_differences(text1: str, text2: str) -> None:
    """
    Shows the differences between two text blocks by printing them side by side
    with differences highlighted.
    
    Args:
        text1: First text block to compare
        text2: Second text block to compare
    """
    lines1 = text1.splitlines()
    lines2 = text2.splitlines()
    
    max_lines = max(len(lines1), len(lines2))
    max_line_length = max(
        max((len(line) for line in lines1), default=0),
        max((len(line) for line in lines2), default=0)
    )
    
    print("\nText Differences:")
    print("-" * (max_line_length * 2 + 10))
    print(f"{'Text 1':<{max_line_length}} | {'Text 2'}")
    print("-" * (max_line_length * 2 + 10))
    
    for i in range(max_lines):
        line1 = lines1[i] if i < len(lines1) else ""
        line2 = lines2[i] if i < len(lines2) else ""
        
        if line1 != line2:
            print(f"\033[91m{line1:<{max_line_length}}\033[0m | \033[91m{line2}\033[0m")
        else:
            print(f"{line1:<{max_line_length}} | {line2}")
    
    print("-" * (max_line_length * 2 + 10))

show_text_differences(text_prompt, text_prompt3)

# print(messages[0]['content'] == batch_messages[0][0]['content'])
# show_text_differences(messages[0]['content'], batch_messages[0][0]['content'])


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

    print(f"x_offset: {x_offset}, y_offset: {y_offset}, scale_factor: {scale_factor}")
    # Create a copy of the background and paste input image
    result = background.copy()
    result.paste(input_image, (x_offset, y_offset))
    
    return result, (x_offset, y_offset), scale_factor

import matplotlib.pyplot as plt

def plot_image(image, x, y, linewidth: int = 2):
    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    plt.grid(False)
    plt.axis('off')
    
    # Draw crosshair lines spanning full image width/height
    width = image.size[0]
    height = image.size[1]
    
    plt.plot([0, width], [y, y], color='black', linewidth=linewidth, linestyle='dotted')
    plt.plot([x, x], [0, height], color='black', linewidth=linewidth, linestyle='dotted')
    plt.plot(x, y, marker='o', color='red', markersize=10)

    print(f"Image size: {image.size}")
    
    plt.show()
# %%
