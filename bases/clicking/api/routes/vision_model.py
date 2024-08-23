from fastapi import APIRouter, Query, File, UploadFile, Form, Path, HTTPException
from clicking.vision_model.core import *
from PIL import Image
import io
import json
import time
from fastapi import Depends

vision_model_router = APIRouter()
vision_model = VisionModel()


@vision_model_router.get("/models", response_model=GetModelsResp, operation_id="get_models")
async def get_models():
    models = vision_model.get_available_models()
    return models

@vision_model_router.get("/model", response_model=GetModelResp, operation_id="get_model")
async def get_model(req: GetModelReq = None):
    
    # Get model info using the TaskType enum
    response  = vision_model.get_model(req.task)
    return response


@vision_model_router.post("/model", operation_id="set_model")
async def set_model(req: SetModelReq = None):
    # Get model info using the TaskType enum
    result = vision_model.set_model(req)
    print(req)
    return {"message": "Model set successfully", "status_code": 200}


@vision_model_router.post("/prediction",operation_id="get_prediction", response_model=PredictionResp)
async def prediction(image: UploadFile = File(...),
    task: TaskType = Form(None),
    input_boxes: str = Form(None),
    input_point: str = Form(None),
    input_label: str = Form(None),
    input_text: str = Form(None),
    ):

    if task is None:
        raise HTTPException(status_code=400, detail="Task is required")

    #Convert to a PIL image
    image_data = await image.read()
    image = Image.open(io.BytesIO(image_data))

    # convert input_boxes to a list of lists
    input_boxes = json.loads(input_boxes) if input_boxes else []

    req = PredictionReq(image=image, task=task, input_point=input_point, input_label=input_label, input_box=input_boxes, input_text=input_text)
    print(req)

    
    response = await vision_model.get_prediction(req)
    return response

# @vision_model_router.post("/auto_annotation",operation_id="get_auto_annotation")
# async def auto_annotation(params: TestReq = Form(...)) -> Dict:

#     # if task is None:
#     #     raise HTTPException(status_code=400, detail="Task is required")

#     print(params.name)

#     # #Convert to a PIL imagex
#     # image_data = await image.read()
#     # image = Image.open(io.BytesIO(image_data))

#     # req = AutoAnnotationReq(image=image,
#     # task=task,
#     # min_mask_region_area = min_mask_region_area,
#     # pred_iou_thresh = pred_iou_thresh,
#     # output_mode = output_mode,
#     # points_per_side = points_per_side,
#     # points_per_batch = points_per_batch,
#     # stability_score_thresh = stability_score_thresh,
#     # stability_score_offset = stability_score_offset,
#     # mask_threshold = mask_threshold,
#     # box_nms_thresh = box_nms_thresh,
#     # crop_n_layers = crop_n_layers,
#     # crop_nms_thresh = crop_nms_thresh,
#     # crop_overlap_ratio = crop_overlap_ratio,
#     # crop_n_points_downscale_factor = crop_n_points_downscale_factor,
#     # # point_grids: Optional[List[np.ndarray]] = None,
#     # use_m2m = use_m2m,
#     # multimask_output = multimask_output,
#     # )
#     # print(req)
#     # response = await vision_model.get_auto_annotation(req)
#     # return response

#     return {"message": "Auto annotation completed successfully", "status_code": 200}



@vision_model_router.post("/auto_annotation", operation_id="get_auto_annotation")
async def auto_annotation(params: AutoAnnotationReq = Depends()) -> Dict:
    print(params.name)
    return {"message": "Auto annotation completed successfully", "status_code": 200}