#%% 
from clicking.prompt_manager.core import PromptManager
import re

PROMPT_PATH = './prompts/prompt_refinement.md'



#%%
template_values = {
    "action": "Pick up chair",
}

prompt_manager = PromptManager(PROMPT_PATH)
prompts =  prompt_manager.get_prompt(type='user', prompt_key='default', template_values=template_values)
print(prompts)

# %%
