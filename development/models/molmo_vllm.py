#%%
from vllm import LLM, SamplingParams
from PIL import Image
import requests
import matplotlib.pyplot as plt
import re
from pydantic import BaseModel, Field
import torch
#%%

class ImagePoint(BaseModel):
    x: float = Field(..., description="X-coordinate of the point as a percentage of the image width")
    y: float = Field(..., description="Y-coordinate of the point as a percentage of the image height")
    alt: str = Field(..., description="Alternative text for the point")

def text_to_image_point(text: str) -> ImagePoint:
    pattern = r'<point x="(\d+(?:\.\d+)?)" y="(\d+(?:\.\d+)?)" alt="([^"]+)">'
    match = re.search(pattern, text)
    
    if not match:
        raise ValueError("Invalid text format. Expected <point x=\"...\" y=\"...\" alt=\"...\">")
    
    x = float(match.group(1))
    y = float(match.group(2))
    alt = match.group(3)
    
    return ImagePoint(x=x, y=y, alt=alt)

def plot_image_with_point(image, point):
    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    plt.grid(False)
    plt.axis('off')

    # Convert the point coordinates to image coordinates
    img_width, img_height = image.size
    x_img = point.x/100 * img_width
    y_img = point.y/100 * img_height
    
    plt.plot(x_img, y_img, marker='*', color='yellow', markersize=15)
    
    plt.show()

#%%
text_input = "Point to the book on top of the cabinet"

image = Image.open('./master_thesis/development/test_received.png') 
llm = LLM(model="allenai/Molmo-7B-D-0924", trust_remote_code=True, dtype=torch.float32, tensor_parallel_size=2)

#%%
sampling_params = SamplingParams(temperature=1.0, max_tokens=200)
outputs = llm.generate({
        "prompt": text_input,
        "multi_modal_data": {"image": image},
    }, sampling_params)

# Print the outputs.

generated_text = outputs[0].outputs[0].text
print(f"Generated text: {generated_text!r}")
# %%
click_point = text_to_image_point(generated_text)
print(click_point)
plot_image_with_point(image, click_point)

# %%
