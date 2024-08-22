#%%
from components.clicking.prompt_manager.core import PromptManager
from PIL import Image
from matplotlib import pyplot as plt
from enum import Enum, auto

# %%
image = Image.open("./datasets/resized_media/gameplay_images/mario_odessey/8.jpg")
plt.grid(False)
plt.axis('off')
plt.imshow(image)

#%%
labeller =  PromptRefiner(prompt_path="./prompts/prompt_refinement.md")

response = labeller.process_prompt(image, "yellow car", PromptMode.EXPANDED_DESCRIPTION)
print(response)
