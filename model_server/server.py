from robyn import Robyn
import os
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import io
import base64
from models.florence2 import Florence2Model
import time


app = Robyn(__file__)
model = Florence2Model()


@app.get("/")
async def h(request):
    return "Hello, world1!"

@app.get("/detection")
async def detection(req):
    req_json = req.json()
    base64_image = req_json['image']
    text_input = req_json['text_input']
    task_prompt = req_json['task_prompt']

    image = Image.open(io.BytesIO(base64.b64decode(base64_image)))

    start_time = time.time()
    results = model.run_inference(image, task_prompt, text_input=text_input)
    end_time = time.time()
    inference_time = end_time - start_time

    response = {
        "bboxes": results[task_prompt]['bboxes'],
        "labels": results[task_prompt]['labels'],
        "inference_time": inference_time
    }
    return response

app.start(port=8082)