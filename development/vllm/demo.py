#%%
from vllm import LLM, SamplingParams

#%%
prompts = [
    "Hello, my name is"
]

sampling_params = SamplingParams(temperature=1.0, top_p=0.95)
llm = LLM(model="allenai/Molmo-7B-D-0924", trust_remote_code=True)

#%%
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
# %%
