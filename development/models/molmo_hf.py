#%%
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image
import requests
import matplotlib.pyplot as plt
import re
from pydantic import BaseModel, Field

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
processor = AutoProcessor.from_pretrained(
    'allenai/Molmo-7B-D-0924',
    trust_remote_code=True,
    torch_dtype='auto',
    device_map='auto'
)

# load the model
model = AutoModelForCausalLM.from_pretrained(
    'allenai/Molmo-7B-D-0924',
    trust_remote_code=True,
    torch_dtype='auto',
    device_map='auto'
)
# %%
# process the image and text
text = "Point to the book on the cabinet"
images = [Image.open('./master_thesis/development/test_received.png')]
inputs = processor.process(
    images=images,
    text=text
)

# plot the image
plt.imshow(images[0])
plt.grid(False)
plt.axis('off')
plt.show()

# move inputs to the correct device and make a batch of size 1
inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}
# %%
import time
start_time = time.time()
output = model.generate_from_batch(
    inputs,
    GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
    tokenizer=processor.tokenizer
)

end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")

# only get generated tokens; decode them to text
generated_tokens = output[0,inputs['input_ids'].size(1):]
generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
click_point = text_to_image_point(generated_text)

#%% 
print(click_point)
plot_image_with_point(images[0], click_point)
