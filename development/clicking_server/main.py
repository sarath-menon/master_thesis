import os
from PIL import Image
import io
import base64
from models.florence2 import Florence2Model
import time
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import models 


app = FastAPI()

# model_id = "/Users/sarathmenon/Documents/master_thesis/clicking_server/Florence-2-base-ft"
# model_id = "/app/Florence-2-base-ft"
florence = models.Florence2Model()
sam2 = models.SAM2Model(type='large')

@app.get("/")
async def h():
    return {"message": "This is a model server for pytorch models from HuggingFace"}

class LocalizationRequest(BaseModel):  
    image: str
    text_input: str
    task_prompt: str

class LocalizationResp(BaseModel):
    bboxes: list
    labels: list
    inference_time: float

class SegmentationRequest(BaseModel):  
    image: str
    input_boxes: list

class SegmentationResp(BaseModel):
    masks: list
    scores: list
    inference_time: float

class AnnotationRequest(BaseModel):  
    image: str
    min_mask_region_area: float
    pred_iou_thresh: float
    stability_score_thresh: float

class AnnotationResp(BaseModel):
    masks: list
    inference_time: float

@app.get("/localization", response_model=LocalizationResp)
async def localization(req: LocalizationRequest):
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

@app.get("/segmentation", response_model=SegmentationResp)
async def segmentation(req: SegmentationRequest):
    base64_image = req.image
    input_boxes  = req.input_boxes 

    # Convert base64 string back to image
    image = base64.b64decode(base64_image)
    image = Image.open(io.BytesIO(image))
    
    masks, scores = sam2.predict_with_batched_bbox(image, input_boxes)

    response = {"masks": masks.tolist(), "scores": scores.tolist(), "inference_time": 0.0}
    return response

@app.get("/annotation", response_model=AnnotationResp)
async def annotate(req: AnnotationRequest):
    base64_image = req.image
    min_mask_region_area = float(req.min_mask_region_area)
    pred_iou_thresh = float(req.pred_iou_thresh)
    stability_score_thresh = float(req.stability_score_thresh)

    # Convert base64 string back to image
    image = base64.b64decode(base64_image)
    image = Image.open(io.BytesIO(image))
    
    masks = sam2.generate_masks(image, min_mask_region_area, pred_iou_thresh)


    response = {"masks": masks, "inference_time": 0.0}
    return response 

# if __name__ == '__main__':
#     uvicorn.run(app, port=8082, host='0.0.0.0')