from clicking.localization.model import Florence2Model
from pydantic import BaseModel
from PIL import Image
import io
import base64
import time

class LocalizationRequest(BaseModel):  
    image: str
    text_input: str
    task_prompt: str

class LocalizationResp(BaseModel):
    bboxes: list
    labels: list
    inference_time: float

florence = Florence2Model()


async def get_localization(req: LocalizationRequest):
    base64_image = req.image
    text_input = req.text_input
    task_prompt = req.task_prompt

    # Convert base64 string back to image
    image = base64.b64decode(base64_image)
    image = Image.open(io.BytesIO(image))
    
    # images_path = "../datasets/resized_media/gameplay_images"
    # image = Image.open(images_path + "/hogwarts_legacy/1.jpg")

    # run inference and measure execution time
    start_time = time.time()
    response = florence.run_inference(image, task_prompt, text_input=text_input)
    end_time = time.time()
    inference_time = end_time - start_time

    response['inference_time'] = inference_time
    return response   