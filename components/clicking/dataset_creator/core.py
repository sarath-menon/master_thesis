import numpy as np
from torchvision import transforms, datasets
from PIL import Image
import matplotlib.pyplot as plt

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

    def sample_dataset(self, indices=None, batch_size=None, show_images=False, show_images_per_batch=4):
        if indices is not None and batch_size is not None:
            raise ValueError("Only one of 'indices' or 'batch_size' should be provided, not both.")
        
        if indices is None and batch_size is None:
            batch_size = 1
        
        if indices is None:
            indices = np.random.choice(len(self.coco_dataset), size=batch_size, replace=False)

        images = []
        class_labels = []
        to_pil = transforms.ToPILImage()
        
        for index in indices:
            image_tensor, annotations = self.coco_dataset[int(index)]
            image = to_pil(image_tensor)
            class_label = self.create_text_input(annotations)
            images.append(image)
            class_labels.append(class_label)

        # plot first 4 images
        if show_images:
            for image, class_label in zip(images, class_labels):
                plt.imshow(image)
                plt.axis(False)
                plt.title(class_label)
                plt.show()

        return images, class_labels

    
