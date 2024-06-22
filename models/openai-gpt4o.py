import os
from openai import AsyncOpenAI
import asyncio
import base64

MODEL = "gpt-4o"
IMAGE_PATH = "screenshot.jpg"

# Open the image file and encode it as a base64 string
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

base64_image = encode_image(IMAGE_PATH)


client = AsyncOpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

async def main():
    stream = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are an expert at QA testing for video games. You will be given a screenshot of a video game and asked to describe what is happening in the game to the user."},
            {"role": "user", "content": [
                {"type": "text", "text": "What is happening in the game?"},
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


asyncio.run(main())