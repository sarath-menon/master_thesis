
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
from .clicking_model import BaseClickingModel
import numpy as np

class GroundingDinoModel(BaseClickingModel):
    def __init__(self):
        self.model_id = "IDEA-Research/grounding-dino-tiny"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.processor = self.load_model(self.model_id)

    def load_model(self, model_id):
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)
        return model, processor

    def run_inference(self, image, task_prompt, text_input=None):
        # task prompt is not required here since model only performs grounded detection
        inputs = self.processor(images=image, text=text_input, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.4,
            text_threshold=0.3,
            target_sizes=[image.size[::-1]]
        )

        response = {
            "bboxes": results[0]["boxes"].numpy().tolist(),
            "labels": results[0]["labels"]
        }
        return response