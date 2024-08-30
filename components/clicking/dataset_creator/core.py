import numpy as np
from torchvision import transforms, datasets
from PIL import Image
import matplotlib.pyplot as plt
from typing import List
from .types import DatasetSample, ImageSample
from clicking.common.pipeline_state import PipelineState

class CocoDataset:
    def __init__(self, data_dir, annFile):
        self.data_dir = data_dir
        self.annFile = annFile
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.coco_dataset = datasets.CocoDetection(root=self.data_dir, annFile=self.annFile, transform=self.transform)
        self.all_object_names = [cat['name'] for cat in self.coco_dataset.coco.cats.values()]
        print(f"Dataset size: {len(self.coco_dataset)}")

    def create_text_input(self, annotations):
        labels = [self.all_object_names[annotation['category_id']] for annotation in annotations]
        text_input = ". ".join(labels) + "." if labels else ""
        return text_input

    def sample_dataset(self, state: PipelineState) -> PipelineState:
        image_samples = []
        to_pil = transforms.ToPILImage()
        
        for index in state.image_ids:
            image_tensor, annotations = self.coco_dataset[int(index)]
            image = to_pil(image_tensor)
            object_name = self.create_text_input(annotations)
            image_samples.append(ImageSample(image=image, object_name=object_name, image_id=index))

        state.dataset_sample = DatasetSample(images=image_samples)
        return state

    def show_images(self, dataset_sample: DatasetSample, show_images_per_batch: int = 4):
        for image_sample in dataset_sample.images[:show_images_per_batch]:
            plt.imshow(image_sample.image)
            plt.axis(False)
            plt.title(image_sample.object_name)
            plt.show()

