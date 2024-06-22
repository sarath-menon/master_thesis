import os
from openai import AsyncOpenAI
import asyncio

import utils

MODEL = "gpt-4o"
IMAGE_PATH = "screenshot_annotated.jpg"
PROMPT_PATH = 'prompts/gpt4-0.md'


client = AsyncOpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

async def main(prompts_dict, base64_image ):
    stream = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": prompts_dict['System']},
            {"role": "user", "content": [
                {"type": "text", "text": prompts_dict['User']},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{base64_image}"}
                }
            ]}
        ],
        stream=True,
        temperature=0.0,
    )
    async for chunk in stream:
        print(chunk.choices[0].delta.content or "", end="")


# Example usage
prompts_dict = utils.markdown_to_dict(PROMPT_PATH)
# print(prompts_dict['User'])
base64_image = utils.encode_image(IMAGE_PATH)
asyncio.run(main(prompts_dict, base64_image))