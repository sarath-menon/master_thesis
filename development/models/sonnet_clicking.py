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
text_prompt = """
Give the outout as JSON in the following format:
reasoning: <your thinking where the object is located in the image>
x: <exact x-coordiante>
y: <exact y-coordiante>
"""

messages = [{"role": "system", "content": 
"""
You are an expert at locating objects on a image. Your task is to help the user locate object in the provided image, by providing the user with the correct coordiantes.
You receive an image from the user and its resolustion and the object you should locate. The output needs to be your best guess of the coordinates of that object.
Give us the middle/center of the object the user wants to locate. Also be precise and don't give rough estimations. Try to be as precicse as possible.
The top left corner is the origin with 0, 0 coordinate.
"""
}]

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
