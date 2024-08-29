import numpy as np
from torchvision import transforms, datasets
from PIL import Image
import matplotlib.pyplot as plt
from typing import List
from .types import DatasetSample

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
        images = []
        class_labels = []
        to_pil = transforms.ToPILImage()
        
        for index in image_ids:
            image_tensor, annotations = self.coco_dataset[int(index)]
            image = to_pil(image_tensor)
            class_label = self.create_text_input(annotations)
            images.append(image)
            class_labels.append(class_label)

        return DatasetSample(images=images, class_labels=class_labels)

    def show_images(self, sample: DatasetSample, show_images_per_batch: int = 4):
        for image, class_label in zip(sample.images[:show_images_per_batch], sample.class_labels[:show_images_per_batch]):
            plt.imshow(image)
            plt.axis(False)
            plt.title(class_label)
            plt.show()

