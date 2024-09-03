
# In[1]:

from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image, ImageDraw, ImageFont 
import requests
import copy
import os
import random
import numpy as np
import PIL

get_ipython().run_line_magic('matplotlib', 'inline')
images_path = "./datasets/resized_media/gameplay_images"

# In[2]:

from transformers.dynamic_module_utils import get_imports
from unittest.mock import patch

def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    """Work around for https://huggingface.co/microsoft/phi-1_5/discussions/72."""
    if not str(filename).endswith("/modeling_florence2.py"):
        return get_imports(filename)
    imports = get_imports(filename)
    imports.remove("flash_attn")
    return imports

# In[6]:

model_id = 'microsoft/Florence-2-base'

with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
        model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)


# In[9]:


colormap = ['blue','orange','green','purple','brown','pink','gray','olive','cyan','red',
            'lime','indigo','violet','aqua','magenta','coral','gold','tan','skyblue']

def run_example(task_prompt, text_input=None):
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    generated_ids = model.generate(
      input_ids=inputs["input_ids"],
      pixel_values=inputs["pixel_values"],
      max_new_tokens=1024,
      early_stopping=False,
      do_sample=False,
      num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height)
    )

    return parsed_answer

import matplotlib.pyplot as plt
import matplotlib.patches as patches
def plot_bbox(image, data, bboxes, labels):
   # Create a figure and axes
    fig, ax = plt.subplots()

    # Display the image
    ax.imshow(image)

    # Plot each bounding box
    for bbox, label in zip(bboxes, labels):
        # Unpack the bounding box coordinates
        x1, y1, x2, y2 = bbox
        # Create a Rectangle patch
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')
        # Add the rectangle to the Axes
        ax.add_patch(rect)
        # Annotate the label
        plt.text(x1, y1, label, color='white', fontsize=8, bbox=dict(facecolor='red', alpha=0.5))

    # Remove the axis ticks and labels
    ax.axis('off')

    # Show the plot
    plt.show()

def convert_to_od_format(data):  
    """  
    Converts a dictionary with 'bboxes' and 'bboxes_labels' into a dictionary with separate 'bboxes' and 'labels' keys.  
  
    Parameters:  
    - data: The input dictionary with 'bboxes', 'bboxes_labels', 'polygons', and 'polygons_labels' keys.  
  
    Returns:  
    - A dictionary with 'bboxes' and 'labels' keys formatted for object detection results.  
    """  
    # Extract bounding boxes and labels  
    bboxes = data.get('bboxes', [])  
    labels = data.get('bboxes_labels', [])  
      
    # Construct the output format  
    od_results = {  
        'bboxes': bboxes,  
        'labels': labels  
    }  
      
    return od_results  

def draw_ocr_bboxes(image, prediction):
    scale = 1
    draw = ImageDraw.Draw(image)
    bboxes, labels = prediction['quad_boxes'], prediction['labels']
    for box, label in zip(bboxes, labels):
        color = random.choice(colormap)
        new_box = (np.array(box) * scale).tolist()
        draw.polygon(new_box, width=3, outline=color)
        draw.text((new_box[0]+8, new_box[1]+2),
                    "{}".format(label),
                    align="right",
        
                    fill=color)
    display(image)

# # OCR
# In[9]:
image = Image.open(images_path + "/captain_toad/1.jpg").convert("RGB")

task_prompt = '<OCR_WITH_REGION>'
results = run_example(task_prompt)
print(results['<OCR_WITH_REGION>']['labels'])
draw_ocr_bboxes(image, results['<OCR_WITH_REGION>'])

# In[ ]:


image = Image.open(images_path + "/captain_toad/2.jpeg").convert("RGB")

task_prompt = '<OCR_WITH_REGION>'
results = run_example(task_prompt)
print(results['<OCR_WITH_REGION>']['labels'])
draw_ocr_bboxes(image, results['<OCR_WITH_REGION>'])


# In[ ]:


image = Image.open(images_path + "/captain_toad/3.jpeg").convert("RGB")

task_prompt = '<OCR_WITH_REGION>'
results = run_example(task_prompt)
print(results['<OCR_WITH_REGION>']['labels'])
draw_ocr_bboxes(image, results['<OCR_WITH_REGION>'])


# In[15]:


images_path = "../datasets/resized_media/gameplay_images"


# # Mario

# In[16]:


image = Image.open(images_path + "/mario_odessey/0.jpg")

task_prompt = '<CAPTION_TO_PHRASE_GROUNDING>'
results = run_example(task_prompt, text_input="pillars.")
print(results)
plot_bbox(image, results['<CAPTION_TO_PHRASE_GROUNDING>'])


# # Hogwarts legacy

# In[7]:


image = Image.open(images_path + "/hogwarts_legacy/2.jpg")

task_prompt = '<OPEN_VOCABULARY_DETECTION>'
results = run_example(task_prompt, text_input=" A circular platform with glowing, mystical symbols, located on the ground beneath the floating book. It emits a faint light.")
print(results)
plot_bbox(image, results[task_prompt], results[task_prompt]['bboxes'], results[task_prompt]['bboxes_labels'])


# ## Unpacking

# In[147]:


image = Image.open(images_path + "/unpacking/0.jpg")

task_prompt = '<CAPTION_TO_PHRASE_GROUNDING>'
results = run_example(task_prompt, text_input="computer. chair. pillow. open box.")
print(results)
plot_bbox(image, results['<CAPTION_TO_PHRASE_GROUNDING>'])


# In[155]:


image = Image.open(images_path + "/unpacking/6.jpg")

task_prompt = '<CAPTION_TO_PHRASE_GROUNDING>'
results = run_example(task_prompt, text_input="oven. trash can.")
print(results)
plot_bbox(image, results['<CAPTION_TO_PHRASE_GROUNDING>'])


# ## Fortnite

# In[37]:


image = Image.open(images_path + "/fortnite/3.jpg")

task_prompt = '<CAPTION_TO_PHRASE_GROUNDING>'
results = run_example(task_prompt, text_input="box.")
print(results)
plot_bbox(image, results['<CAPTION_TO_PHRASE_GROUNDING>'])


# In[45]:


image = Image.open(images_path + "/fortnite/4.jpg")

task_prompt = '<CAPTION_TO_PHRASE_GROUNDING>'
results = run_example(task_prompt, text_input="barrel.")
print(results)
plot_bbox(image, results['<CAPTION_TO_PHRASE_GROUNDING>'])

