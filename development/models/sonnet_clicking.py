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

text_prompt = """
Analyze the image and provide the following JSON output:
{
    "reasoning": "Explanation of how you identified the object's location in 20 words",
    "x": "Exact pixel x-coordinate of object's center (integer)",
    "y": "Exact pixel y-coordinate of object's center (integer)"
}
"""

image = Image.open("./datasets/anthropic_clicking/anthropic_demo.jpg")
response = await client.get_image_response(image=image, messages=messages, text_prompt=text_prompt)

response = json.loads(response)
response 
#%%
import matplotlib.pyplot as plt

def plot_image_with_point(image, x, y):
    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    plt.grid(False)
    plt.axis('off')

    plt.plot(x, y, marker='*', color='yellow', markersize=15)
    
    plt.show()

plot_image_with_point(image, response["x"], response["y"])
# %%
