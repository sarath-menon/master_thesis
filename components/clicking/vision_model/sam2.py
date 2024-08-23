import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from fastapi import HTTPException
from clicking.vision_model.types import TaskType, PredictionReq, SegmentationResp, PredictionResp
from pycocotools import mask as mask_utils
from typing import Dict, Any
from clicking.vision_model.utils import coco_encode_rle

class SAM2:
    variant_to_id = {
        "sam2_hiera_tiny": "facebook/sam2-hiera-tiny" ,
        "sam2_hiera_base": "facebook/sam2-hiera-base" ,
        "sam2_hiera_large": "facebook/sam2-hiera-large" ,
    }

    task_prompts = {TaskType.SEGMENTATION_WITH_CLICKPOINT: "", TaskType.SEGMENTATION_WITH_BBOX: "", TaskType.SEGMENTATION_WITH_BBOX: ""}

    def __init__(self, variant="sam2_hiera_tiny"):
        self.name = 'sam2'
        self.variant = variant

        # select the device for computation
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        print(f"Using {self.device} for {self.name}")

        self.predictor = self.load_predictor_from_hub(self.variant)

        if self.device.type == "cuda":
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            # turn on tfloat32 for Ampere GPUs 
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        elif self.device.type == "mps":
            print(
                "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
                "give numerically different outputs and sometimes degraded performance on MPS. "
                "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
            )

        self.task_to_method = {
            TaskType.SEGMENTATION_WITH_CLICKPOINT: self.predict_with_clickpoint,
            TaskType.SEGMENTATION_WITH_BBOX: self.predict_with_bbox,
            TaskType.SEGMENTATION_WITH_CLICKPOINT_AND_BBOX: self.predict_with_clickpoint_and_bbox
        }


    @staticmethod
    def variants():
        return list(SAM2.variant_to_id.keys())
    
    @staticmethod
    def tasks():
        return list(SAM2.task_prompts.keys())

    def load_predictor_from_disk(self, variant):
        if variant not in self.variant_to_id:
            raise HTTPException(status_code=400, detail=f"Invalid variant: {variant}. Please choose from: {list(self.variant_to_id.keys())}")

        self.checkpoint = "./checkpoints/sam2/" + self.variant_to_id[variant]["checkpoint"]
        self.model_cfg = self.variant_to_id[variant]["model_cfg"]
        model = build_sam2(self.model_cfg, self.checkpoint, device=self.device)
        return SAM2ImagePredictor(model)

    def load_predictor_from_hub(self, variant):
        if variant not in self.variant_to_id:
            raise HTTPException(status_code=400, detail=f"Invalid variant: {variant}. Please choose from: {list(self.variant_to_id.keys())}")

        return SAM2ImagePredictor.from_pretrained(self.variant_to_id[variant])
        

    def predict(self, req: PredictionReq) -> PredictionResp:
        if req.task not in self.task_to_method:
            raise ValueError(f"Invalid task type: {req.task}")
        elif req.image is None:
            raise ValueError("Image is required for any vision task")
        
        predict_method = self.task_to_method[req.task]
        response = predict_method(req)
        return PredictionResp(prediction=response)

    # Modify existing methods to return results instead of showing them
    def predict_with_clickpoint(self, req: PredictionReq):
        if req.click_point is None:
            raise ValueError("Click point is required for clickpoint task")

        self.predictor.set_image(req.image)
        masks, scores, logits = self.predictor.predict(
            point_coords=req.click_point,
            point_labels=req.click_label,
            multimask_output=True,
        )
        sorted_ind = np.argsort(scores)[::-1]
        return masks[sorted_ind], scores[sorted_ind]

    def predict_with_bbox(self, req: PredictionReq) -> SegmentationResp:
        if req.input_box is None:
            raise ValueError("Bbox is required for bbox task")

        # Convert input_box to numpy array
        input_box = np.array(req.input_box)


        self.predictor.set_image(req.image)
        masks, scores, _ = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False,
        )

        scores = scores.tolist()
        masks = [coco_encode_rle(mask) for mask in masks]

        response = SegmentationResp(masks=masks, scores=scores)
        return response

    def predict_with_clickpoint_and_bbox(self, req: PredictionReq):
        if req.click_point is None:
            raise ValueError("Click point is required for clickpoint task")
        if req.bbox is None:
            raise ValueError("Bbox is required for bbox task")

        self.predictor.set_image(req.image)
        masks, scores, logits = self.predictor.predict(
            point_coords=req.click_point,
            point_labels=req.click_label,
            box=req.bbox[None, :],
            multimask_output=False,
        )
        return masks, scores, logits

    def generate_masks(self, image, min_mask_region_area, pred_iou_thresh):
        mask_generator = SAM2AutomaticMaskGenerator(self.sam2_model, min_mask_region_area=min_mask_region_area,
        pred_iou_thresh=pred_iou_thresh,
        output_mode='coco_rle')
        image_np = np.array(image.convert("RGB"))
        masks = mask_generator.generate(image_np)
        return masks