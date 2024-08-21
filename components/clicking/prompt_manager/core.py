import re


class PromptManager:
    def __init__(self, file_path):
        self.file_path = file_path
    
    def load_prompts(self, template_values=None):
        with open(self.file_path, 'r') as file:
            content = file.read()

        pattern = r'^(#)\s*(.*?)\s*\n(.*?)(?=\n#\s|\Z)'
        matches = re.findall(pattern, content, re.S | re.M)
        
        system_prompt = None
        user_prompts = {}

        for level, heading, text in matches:
            heading = heading.strip()
            text = text.strip()
            if level != '#':
                break

            if heading == 'System prompt':
                system_prompt = text
            elif heading == 'User prompt':
                user_prompts.update(self.parse_user_prompts(text))
            else:
                raise ValueError(f"Heading '{heading}' not found in prompt dictionary.")

        # Apply template values if provided
        if template_values:      
            for prompt_key, prompt_content in user_prompts.items():
                user_prompts[prompt_key] = prompt_content.format(**template_values)

        prompts = {
            "system_prompt": system_prompt,
            "user_prompts": user_prompts
        }

        return prompts

    def parse_user_prompts(self, content):
        pattern = r'^(##)\s*(.*?)\s*\n(.*?)(?=\n##\s|\Z)'
        matches = re.findall(pattern, content, re.S | re.M)
        prompts = {}
        for _, heading, text in matches:
            heading = heading.strip()
            if heading in prompts:
                raise ValueError(f"Duplicate subheading '{heading}' found.")
            prompts[heading] = text.strip()
        return prompts
    

    def get_prompt(self, type=None, prompt_key=None, template_values=None):
        prompt_dict = self.load_prompts(template_values=template_values)

        if type == 'system':
            return prompt_dict['system_prompt']
        elif type == 'user':
            if prompt_key is None:
                raise ValueError("Prompt key is required for user prompts.")
            elif prompt_key in prompt_dict['user_prompts']:
                return prompt_dict['user_prompts'][prompt_key]
            else:
                raise ValueError(f"User prompt '{prompt_key}' not found in prompt dictionary.")

        else:
            raise ValueError(f"Prompt type '{type}' not found in prompt dictionary.")