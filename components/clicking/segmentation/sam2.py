import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from fastapi import HTTPException

class SAM2:
    variant_to_id = {
        "sam2_hiera_tiny": {"checkpoint": "sam2_hiera_tiny.pt", "model_cfg": "sam2_hiera_t.yaml"},
        "sam2_hiera_base": {"checkpoint": "sam2_hiera_base.pt", "model_cfg": "sam2_hiera_b.yaml"},
        "sam2_hiera_large": {"checkpoint": "sam2_hiera_large.pt", "model_cfg": "sam2_hiera_l.yaml"}
    }

    task_prompts = [ "with_clickpoint_and_bbox", "with_batched_bbox"]

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

        self.checkpoint = "./checkpoints/sam2/" + self.variant_to_id[self.variant]["checkpoint"]
        self.model_cfg = self.variant_to_id[self.variant]["model_cfg"]
        self.sam2_model = build_sam2(self.model_cfg, self.checkpoint, device=self.device)
        self.predictor = SAM2ImagePredictor(self.sam2_model)

    @staticmethod
    def variants():
        return list(SAM2.variant_to_id.keys())
    
    @staticmethod
    def tasks():
        return SAM2.task_prompts

    def load_model(self, variant):
        if variant not in self.variant_to_id:
            raise HTTPException(status_code=400, detail=f"Invalid variant: {variant}. Please choose from: {list(self.variant_to_id.keys())}")

        self.checkpoint = "./checkpoints/sam2/" + self.variant_to_id[variant]["checkpoint"]
        self.model_cfg = self.variant_to_id[variant]["model_cfg"]
        return build_sam2(self.model_cfg, self.checkpoint, device=self.device)

    def predict_with_clickpoint(self, image, input_point, input_label):
        self.predictor.set_image(image)
        self.show_points(input_point, input_label, plt.gca())
        self.show_image(image)
        masks, scores, logits = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        scores = scores[sorted_ind]
        logits = logits[sorted_ind]
        self.show_masks(image, masks, scores, point_coords=input_point, input_labels=input_label, borders=True)

    def predict_with_bbox(self, image, input_box):
        self.predictor.set_image(image)
        masks, scores, _ = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False,
        )
        return masks, scores

    def predict_with_batched_bbox(self, image, input_boxes):
        self.predictor.set_image(image)
        masks, scores, _ = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )
        return masks, scores
        

    def predict_with_clickpoint_and_bbox(self, image, input_point, input_label, input_box):
        self.predictor.set_image(image)
        masks, scores, logits = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            box=input_box,
            multimask_output=False,
        )
        self.show_masks(predictor.image, masks, scores, box_coords=input_box, point_coords=input_point, input_labels=input_label)

    def batched_prediction_multiple_images(self, img_batch, boxes_batch):
        self.predictor.set_image_batch(img_batch)
        masks_batch, scores_batch, _ = self.predictor.predict_batch(
            None,
            None,
            box_batch=boxes_batch,
            multimask_output=False,
            output_mode='coco_rle'
        )
        return masks_batch, scores_batch

    def generate_masks(self, image, min_mask_region_area, pred_iou_thresh):
        mask_generator = SAM2AutomaticMaskGenerator(self.sam2_model, min_mask_region_area=min_mask_region_area,
        pred_iou_thresh=pred_iou_thresh,
        output_mode='coco_rle')
        image_np = np.array(image.convert("RGB"))
        masks = mask_generator.generate(image_np)
        return masks