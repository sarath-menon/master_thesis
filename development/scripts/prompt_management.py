#%% 
from promptdown import StructuredPrompt
PROMPT_PATH = './prompts/instruction_refinement.prompt.md'

#%% load prompt by heading

structured_prompt = StructuredPrompt.from_promptdown_file(PROMPT_PATH)
print(structured_prompt)

#%%

# Load your structured prompt from a file or string that contains template placeholders
structured_prompt = StructuredPrompt.from_promptdown_file(PROMPT_PATH)

# Define the template values to apply
template_values = {
    "topic": "Python programming",
    "concept": "decorators"
}

# Apply the template values
structured_prompt.apply_template_values(template_values)

# Output the updated prompt
print(structured_prompt.system_message)