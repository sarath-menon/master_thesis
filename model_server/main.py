import os
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import io
import base64
from models.florence2 import Florence2Model
import time
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
PORT = 8082
model = Florence2Model()

@app.get("/")
async def h():
    return "Hello, world1!"

class Request(BaseModel):
    image: str
    text_input: str
    task_prompt: str

class Response(BaseModel):
    bboxes: list
    labels: list
    inference_time: float

@app.get("/detection", response_model=Response)
async def detection(req: Request):
    base64_image = req.image
    text_input = req.text_input
    task_prompt = req.task_prompt

    image = Image.open(io.BytesIO(base64.b64decode(base64_image)))

    # run inference and measure execution time
    start_time = time.time()
    response = model.run_inference(image, task_prompt, text_input=text_input)
    end_time = time.time()
    inference_time = end_time - start_time

    response['inference_time'] = inference_time
    return response   
