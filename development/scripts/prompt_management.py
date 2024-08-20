#%% 
from promptdown import StructuredPrompt

PROMPT_PATH = './prompts/instruction_refinement.md'

import re

#%%
class PromptManager:
    def __init__(self, file_path):
        self.file_path = file_path

    def get_content_by_heading(self, content):
        pattern = r'^(#)\s*(.*?)\s*\n(.*?)(?=\n#\s|\Z)'
        matches = re.findall(pattern, content, re.S | re.M)
        
        system_prompt = None
        user_prompts = {}

        for level, heading, text in matches:
            if level != '#':  # Level 1 heading
                break
        
            print(heading)
            if heading.strip() == 'System prompt':
                system_prompt = text.strip()
            elif heading.strip() == 'User prompt':
                content = text.strip()
                # user_prompts[heading.strip()] = content
                level_2_prompts = []

                # Pattern to match headings of all levels and their content
                pattern = r'^(##)\s*(.*?)\s*\n(.*?)(?=\n##\s|\Z)'
                matches = re.findall(pattern, content, re.S | re.M)

                for level, heading, content in matches:
                    user_prompts[heading.strip()] = content.strip()
            else:
                raise ValueError(f"Heading '{heading}' not found in prompt dictionary.")

        result = {
            "system_prompt": system_prompt,
            "user_prompts": user_prompts
        }

        return result

    def load_prompts(self, template_values=None):
        with open(self.file_path, 'r') as file:
            content = file.read()
            prompts = self.get_content_by_heading(content)

        # Apply template values if provided
        # Apply template values if provided
        if template_values:      
            for prompt_key, prompt_content in prompts['user_prompts'].items():
                prompts['user_prompts'][prompt_key] = prompt_content.format(**template_values)

        return prompts

    

    def get_prompt(self, heading=None, template_values=None):
        prompt_dict = self.load_prompts(template_values=template_values)
        print(prompt_dict)

        # if heading is None:
        #     return prompt_dict
        # elif heading in prompt_dict:
        #     return prompt_dict[heading]
        # else:
        #     raise ValueError(f"Heading '{heading}' not found in prompt dictionary.")
        return prompt_dict
        
#%%
template_values = {
    "action": "Pick up chair",
}

prompt_manager = PromptManager(PROMPT_PATH)
prompts =  prompt_manager.get_prompt(heading='User prompt', template_values=template_values)
print(prompts['user_prompts']['default'])


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