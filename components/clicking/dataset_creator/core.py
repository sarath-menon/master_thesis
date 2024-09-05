#%%
import numpy as np
from torchvision import transforms, datasets
from PIL import Image
import matplotlib.pyplot as plt
from typing import List
from clicking.common.data_structures import ClickingImage, ImageObject, ObjectCategory
from pycocotools import mask as mask_utils
from clicking.common.mask import SegmentationMask, SegmentationMode
from clicking.common.bbox import BoundingBox, BBoxMode
import uuid

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

    def length(self):
        return len(self.coco_dataset)

    def sample_dataset(self, image_ids: List[int]) -> List[ClickingImage]:
        clicking_images = []
        to_pil = transforms.ToPILImage()
        
        for index in image_ids:
            image_tensor, annotations = self.coco_dataset[int(index)]
            image = to_pil(image_tensor)
            objects = self._create_image_objects(annotations)
            clicking_images.append(ClickingImage(image=image, id=str(index), annotated_objects=objects))

        return clicking_images

    def _create_image_objects(self, annotations) -> List[ImageObject]:
        objects = []
        for ann in annotations:
            category = self._get_object_category(ann['category_id'])
            bbox = BoundingBox(bbox=ann['bbox'], mode=BBoxMode.XYWH)
            
            # Convert polygon to RLE
            if isinstance(ann['segmentation'], list):
                # Get image dimensions
                img_info = self.coco_dataset.coco.loadImgs(ann['image_id'])[0]
                height, width = img_info['height'], img_info['width']
                
                # Convert polygon to RLE
                rle = mask_utils.frPyObjects(ann['segmentation'], height, width)
                if len(rle) == 1:
                    rle = rle[0]
                else:
                    rle = mask_utils.merge(rle)
            else:
                # If it's already in RLE format, use it as is
                rle = ann['segmentation']
            
            mask = SegmentationMask(coco_rle=rle, mode=SegmentationMode.COCO_RLE)
            
            obj = ImageObject(
                name=self.all_object_names[ann['category_id']],
                category=category,
                bbox=bbox,
                mask=mask
            )
            objects.append(obj)

        return objects

    def _get_object_category(self, category_id: int) -> ObjectCategory:
        # This is a placeholder. You'll need to implement the logic to map
        # COCO categories to your ObjectCategory enum
        return ObjectCategory.GAME_ASSET  # Default to GAME_ASSET for now

    def show_images(self, clicking_images: List[ClickingImage], show_images_per_batch: int = 4):
        for clicking_image in clicking_images[:show_images_per_batch]:
            plt.imshow(clicking_image.image)
            plt.axis(False)
            plt.title(", ".join([obj.name for obj in clicking_image.annotated_objects]))
            plt.show()

    def get_ground_truth(self, image_id: int):
        img_info = self.coco_dataset.coco.loadImgs(image_id)[0]
        ann_ids = self.coco_dataset.coco.getAnnIds(imgIds=image_id)
        anns = self.coco_dataset.coco.loadAnns(ann_ids)

        objects = []

        for ann in anns:
            if 'segmentation' in ann:
                if isinstance(ann['segmentation'], dict):
                    # RLE format
                    rle = ann['segmentation']
                elif isinstance(ann['segmentation'], list):
                    # Polygon format
                    rle = mask_utils.frPyObjects(ann['segmentation'], img_info['height'], img_info['width'])
                    if len(rle) == 1:
                        rle = rle[0]
                    else:
                        rle = mask_utils.merge(rle)
                else:
                    print(f"Skipping annotation with invalid segmentation format: {ann['segmentation']}")
                    continue
                
                category = self._get_object_category(ann['category_id'])
                bbox = BoundingBox(ann['bbox'], mode=BBoxMode.XYWH)
                
                # Ensure the RLE is in the correct format
                if isinstance(rle, dict) and 'counts' in rle and 'size' in rle:
                    mask = SegmentationMask(coco_rle=rle, mode=SegmentationMode.COCO_RLE)
                else:
                    # If not in the correct format, skip this annotation
                    continue
                
                obj = ImageObject(
                    name=self.all_object_names[ann['category_id'] - 1],
                    description="",
                    category=category,
                    bbox=bbox,
                    mask=mask
                )
                objects.append(obj)

        return objects

#%% Demo code
if __name__ == "__main__":
    from clicking.vision_model.visualization import show_segmentation_predictions
    from clicking.common.mask import SegmentationMask, SegmentationMode
    from clicking.common.bbox import BoundingBox, BBoxMode
    from clicking.common.data_structures import ImageWithDescriptions
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

# %%
