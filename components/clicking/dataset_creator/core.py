#%%
import numpy as np
from torchvision import transforms, datasets
from PIL import Image
import matplotlib.pyplot as plt
from typing import List
from clicking.dataset_creator.types import DatasetSample, ImageSample
from pycocotools import mask as mask_utils
from clicking.vision_model.mask import SegmentationMask, SegmentationMode

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
        labels = [self.all_object_names[annotation['category_id'] - 1] for annotation in annotations]
        text_input = ". ".join(labels) + "." if labels else ""
        return text_input

    def sample_dataset(self, image_ids: List[int]) -> DatasetSample:
        image_samples = []
        to_pil = transforms.ToPILImage()
        
        for index in image_ids:
            image_tensor, annotations = self.coco_dataset[int(index)]
            image = to_pil(image_tensor)
            object_name = self.create_text_input(annotations)
            image_samples.append(ImageSample(image=image, object_name=object_name, image_id=index))

        return DatasetSample(images=image_samples)

    def show_images(self, dataset_sample: DatasetSample, show_images_per_batch: int = 4):
        for image_sample in dataset_sample.images[:show_images_per_batch]:
            plt.imshow(image_sample.image)
            plt.axis(False)
            plt.title(image_sample.object_name)
            plt.show()

    def get_ground_truth(self, image_id: int):
        img_info = self.coco_dataset.coco.loadImgs(image_id)[0]
        ann_ids = self.coco_dataset.coco.getAnnIds(imgIds=image_id)
        anns = self.coco_dataset.coco.loadAnns(ann_ids)

        masks = []
        class_labels = []

        for ann in anns:
            if 'segmentation' in ann:
                if isinstance(ann['segmentation'], dict):
                    # RLE format
                    rle = ann['segmentation']
                else:
                    # Polygon format
                    rle = mask_utils.frPyObjects(ann['segmentation'], img_info['height'], img_info['width'])
                # Convert RLE to binary mask
                binary_mask = mask_utils.decode(rle)
                masks.append(binary_mask)
                class_labels.append(self.all_object_names[ann['category_id'] - 1])

        return masks, class_labels

#%% Demo code
if __name__ == "__main__":
    from clicking.vision_model.visualization import show_segmentation_predictions
    from clicking.vision_model.mask import SegmentationMask, SegmentationMode
    from clicking.vision_model.types import SegmentationResults
    from clicking.prompt_refinement.core import ImageWithDescriptions
    import random

    data_dir = "./datasets/label_studio_gen/coco_dataset/images"
    annFile = "./datasets/label_studio_gen/coco_dataset/result.json"

    coco_dataset = CocoDataset(data_dir, annFile)

    # Sample a random image
    random_image_id = random.choice(range(len(coco_dataset.coco_dataset)))
    
    # Use sample_dataset to get the image
    dataset_sample = coco_dataset.sample_dataset([random_image_id])
    image_sample = dataset_sample.images[0]
    image = image_sample.image
    object_name = image_sample.object_name

    # Get ground truth
    masks, class_labels = coco_dataset.get_ground_truth(random_image_id)

    # Convert masks to SegmentationMask type
    segmentation_masks = []
    for mask, label in zip(masks, class_labels):
        rle = mask_utils.encode(np.asfortranarray(mask))
        segmentation_mask = SegmentationMask(rle, mode=SegmentationMode.COCO_RLE, object_name=label, description="")
        segmentation_masks.append(segmentation_mask)

    # Create SegmentationResults object
    processed_sample = ImageWithDescriptions(image=image, image_id=str(random_image_id), object_name=object_name)
    segmentation_results = SegmentationResults(
        processed_samples=[processed_sample],
        predictions={str(random_image_id): segmentation_masks}
    )

    # Show the segmentation predictions
    show_segmentation_predictions(segmentation_results)

    # Print the class labels
    print("Class labels:")
    for label in class_labels:
        print(f"- {label}")

# %%
