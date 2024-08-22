#%%
from litellm import acompletion, completion
import asyncio
import os
import dotenv

dotenv.load_dotenv()

## set ENV variables
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
# %% OpenAI text generation

messages = [{ "content": "Hello, how are you?","role": "user"}]

response = completion(model="gpt-3.5-turbo", messages=messages, stream=True)
for part in response:
    print(part.choices[0].delta.content or "", end="", flush=True)

# %% openai VLM

response = completion(
    model = "gpt-4o", 
    stream=True,
    messages=[
        {
            "role": "user",
            "content": [
                            {
                                "type": "text",
                                "text": "What’s in this image?"
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
                                }
                            }
                        ]
        }
    ],
)
#%%
response.choices[0].message.content