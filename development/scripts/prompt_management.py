#%% 
from promptdown import StructuredPrompt

PROMPT_PATH = './prompts/instruction_refinement.md'

import re

#%%
class PromptManager:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_prompts(self, template_values=None):
        with open(self.file_path, 'r') as file:
            content = file.read()

        # to match markdown headings and their corresponding text
        pattern = r'^(#{1,6})\s*(.*?)\s*\n(.*?)(?=\n#{1,6}\s|\Z)'
        matches = re.findall(pattern, content, re.S | re.M)  

        result = {}
        for match in matches:
            level, heading, text = match

            # check and fill in template values
            if template_values:
                text = text.format(**template_values)

            result[heading.strip()] = text.strip()

        return result

    def get_prompt(self, heading=None, template_values=None):
        prompt_dict = self.load_prompts(template_values=template_values)

        if heading is None:
            return prompt_dict
        elif heading in prompt_dict:
            return prompt_dict[heading]
        else:
            raise ValueError(f"Heading '{heading}' not found in prompt dictionary.")
        
#%%
template_values = {
    "action": "Pick up chair",
}

prompt_manager = PromptManager(PROMPT_PATH)
prompts =  prompt_manager.get_prompt(heading='User prompt', template_values=template_values)

print(prompts)


#%%

# Load your structured prompt from a file or string that contains template placeholders
structured_prompt = StructuredPrompt.from_promptdown_string(promptdown_string)

# Define the template values to apply
template_values = {
    "topic": "Python programming",
    "concept": "decorators"
}

# Apply the template values
structured_prompt.apply_template_values(template_values)

# Output the updated prompt
print(structured_prompt)