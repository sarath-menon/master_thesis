#%%
import litellm
import os
from clicking.vision_model.utils import pil_to_base64

os.environ["RUNPOD_API_KEY"] = "HF8ASYIZBGNGFD0JIOV2M771XXV79APXIUVHUL5O"
RUNPOD_ENDPOINT_ID = "vllm-66twa2o3hpknvq"
RUNPOD_ENDPOINT_URL = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/openai/v1"

#%%
from PIL import Image
text_input = "Point to the button named Accept"


def get_response(text_input, image):
    base64_image = pil_to_base64(image)
    response = litellm.completion(
        model="openai/allenai/Molmo-7B-D-0924",               
    api_key=os.environ["RUNPOD_API_KEY"],                
    api_base=RUNPOD_ENDPOINT_URL,     
    temperature=0.0,
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text_input},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                },
            ],
        }
        ],
    )
    return response

#%%
import time

image = Image.open('./datasets/resized_media/monopoly_images/1.jpg') 


start_time = time.time()
response = get_response(text_input, image)
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")
print(response.choices[0].message.content)