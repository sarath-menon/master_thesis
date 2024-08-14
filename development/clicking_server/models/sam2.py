import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

class SAM2Model:
    def __init__(self, type="tiny"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        model_type_table = {"tiny": {"checkpoint": "sam2_hiera_tiny.pt", "model_cfg": "sam2_hiera_t.yaml"},
                     "base": {"checkpoint": "sam2_hiera_base.pt", "model_cfg": "sam2_hiera_b.yaml"},
                     "large": {"checkpoint": "sam2_hiera_large.pt", "model_cfg": "sam2_hiera_l.yaml"}}

        sam2_checkpoint = "./checkpoints/" + model_type_table[type]["checkpoint"]
        model_cfg = model_type_table[type]["model_cfg"]
        self.sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")
        self.predictor = SAM2ImagePredictor(self.sam2_model)

    def load_model(self, model_cfg, checkpoint):
        return build_sam2(model_cfg, checkpoint, device="cuda")

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
            multimask_output=False
        )
        return masks_batch, scores_batch

    def generate_masks(self, image, min_mask_region_area, pred_iou_thresh):
        mask_generator = SAM2AutomaticMaskGenerator(self.sam2_model, min_mask_region_area=min_mask_region_area,
        pred_iou_thresh=pred_iou_thresh,
        output_mode='coco_rle')
        image_np = np.array(image.convert("RGB"))
        masks = mask_generator.generate(image_np)
        return masks