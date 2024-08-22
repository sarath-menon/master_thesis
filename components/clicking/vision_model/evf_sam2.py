#%%

from transformers import AutoTokenizer
import torch
import numpy as np 
import sys

import time
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, BitsAndBytesConfig
from evf_sam.model.segment_anything.utils.transforms import ResizeLongestSide
from evf_sam.model.evf_sam2 import EvfSam2Model
from clicking.vision_model.types import TaskType, PredictionReq, SegmentationResp, PredictionResp
from clicking.vision_model.utils import coco_encode_rle

#%%

class EVF_SAM:
    variant_to_id = {
        "sam2": "sam2"
    }
    task_prompts = {TaskType.SEGMENTATION_WITH_TEXT: ""}

    def __init__(self, version='./EVF-SAM/checkpoints', variant='sam2'):
        self.name = 'evf_sam2'
        self.variant = variant
        self.version = version
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.version,
            padding_side="right",
            use_fast=False,
        )
        self.kwargs = {
            "torch_dtype": torch.half,
        }
        self.model = self.load_model(variant='sam2')
    
    @staticmethod
    def variants():
        return list(EVF_SAM.variant_to_id.keys())
    
    @staticmethod
    def tasks():
        return list(EVF_SAM.task_prompts.keys())
       
    def load_model(self, variant):
        if variant not in self.variant_to_id:
            raise HTTPException(status_code=400, detail=f"Invalid variant: {variant}. Please choose from: {list(self.variant_to_id.keys())}")
            
        model = EvfSam2Model.from_pretrained(self.version, low_cpu_mem_usage=True, **self.kwargs)
        del model.visual_model.memory_encoder
        del model.visual_model.memory_attention
        model = model.cuda().eval()
        return model

    def sam_preprocess(
        self,
        x: np.ndarray,
        pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
        pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
        img_size=1024) -> torch.Tensor:
        '''
        preprocess of Segment Anything Model, including scaling, normalization and padding.  
        preprocess differs between SAM and Effi-SAM, where Effi-SAM use no padding.
        input: ndarray
        output: torch.Tensor
        '''
        assert img_size==1024, \
            "both SAM and Effi-SAM receive images of size 1024^2, don't change this setting unless you're sure that your employed model works well with another size."
        x = ResizeLongestSide(img_size).apply_image(x)
        resize_shape = x.shape[:2]
        x = torch.from_numpy(x).permute(2,0,1).contiguous()

        # Normalize colors
        x = (x - pixel_mean) / pixel_std
        x = F.interpolate(x.unsqueeze(0), (img_size, img_size), mode="bilinear").squeeze(0)
        return x, resize_shape

    def beit3_preprocess(self, x: np.ndarray, img_size=224) -> torch.Tensor:
        '''
        preprocess for BEIT-3 model.
        input: ndarray
        output: torch.Tensor
        '''
        beit_preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC), 
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        return beit_preprocess(x)

    @torch.no_grad()
    def predict(self, req: PredictionReq) -> PredictionResp:
        if req.task not in self.task_prompts:
            raise ValueError(f"Invalid task type: {req.task}")
        elif req.image is None:
            raise ValueError("Image is required for any vision task")
        elif req.input_text is None:
            raise ValueError("Text input is required for evf_sam2")

        image_np = np.array(req.image)
        original_size_list = [image_np.shape[:2]]

        image_beit = self.beit3_preprocess(image_np, 224).to(dtype=self.model.dtype, device=self.model.device)

        image_sam, resize_shape = self.sam_preprocess(image_np)
        image_sam = image_sam.to(dtype=self.model.dtype, device=self.model.device)

        input_ids = self.tokenizer(req.input_text, return_tensors="pt")["input_ids"].to(device=self.model.device)

        pred_mask = self.model.inference(
            image_sam.unsqueeze(0),
            image_beit.unsqueeze(0),
            input_ids,
            resize_list=[resize_shape],
            original_size_list=original_size_list,
        )
        mask = pred_mask.detach().cpu().numpy()[0]
        mask = mask > 0

        masks = [coco_encode_rle(mask)]
        return PredictionResp(prediction=SegmentationResp(masks=masks))
#%%

# version = "./EVF-SAM/checkpoints"
# model = EVF_SAM(version)

# image = Image.open("./EVF-SAM/assets/zebra.jpg")
# req = PredictionReq(image=image, task=TaskType.SEGMENTATION_WITH_TEXT, input_text="zebra")
# masks = model.predict(req)

