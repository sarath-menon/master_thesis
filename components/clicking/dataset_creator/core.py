import numpy as np
from torchvision import transforms, datasets
from PIL import Image
import matplotlib.pyplot as plt
from typing import List
from .types import DatasetSample, DataSample

class CocoDataset:
    def __init__(self, data_dir, annFile):
        self.data_dir = data_dir
        self.annFile = annFile
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.coco_dataset = datasets.CocoDetection(root=self.data_dir, annFile=self.annFile, transform=self.transform)
        self.all_class_labels = [cat['name'] for cat in self.coco_dataset.coco.cats.values()]
        print(f"Dataset size: {len(self.coco_dataset)}")

    def create_text_input(self, annotations):
        labels = [self.all_class_labels[annotation['category_id']] for annotation in annotations]
        text_input = ". ".join(labels) + "." if labels else ""
        return text_input

    def sample_dataset(self, image_ids: List[int]) -> DatasetSample:
        samples = []
        to_pil = transforms.ToPILImage()
        
        for index in image_ids:
            image_tensor, annotations = self.coco_dataset[int(index)]
            image = to_pil(image_tensor)
            class_label = self.create_text_input(annotations)
            samples.append(DataSample(image=image, class_label=class_label))

        return DatasetSample(samples=samples)

    def show_images(self, dataset_sample: DatasetSample, show_images_per_batch: int = 4):
        for sample in dataset_sample.samples[:show_images_per_batch]:
            plt.imshow(sample.image)
            plt.axis(False)
            plt.title(sample.class_label)
            plt.show()

