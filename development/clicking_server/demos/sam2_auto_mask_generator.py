# %%
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
np.random.seed(3)
import matplotlib.pyplot as plt
import io
import base64
from PIL import Image
import requests
from io import BytesIO
from matplotlib import patches

#%%
class ClickingDatasetAPI:
    def __init__(self, server_url="http://localhost:8082"):
        self.server_url = server_url
        self.annotation_endpoint = f"{server_url}/annotation"

    def _encode_image_to_base64(self, image):
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def get_masks(self, image, min_mask_region_area=100.0, pred_iou_thresh=0.94, stability_score_thresh=0.7):
        img_str = self._encode_image_to_base64(image)
        payload = {
            "image": img_str,
            "min_mask_region_area": min_mask_region_area,
            "pred_iou_thresh": pred_iou_thresh,
            "stability_score_thresh": stability_score_thresh
        }
        response = requests.get(self.annotation_endpoint, json=payload)
        return response.json()


def plot_bboxes_on_image(image, masks):
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    for mask in masks:
        bbox = mask['bbox']
        x1, y1, width, height = bbox

        rect = patches.Rectangle((x1, y1), width, height, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.axis('off')
    plt.show()

plot_bboxes_on_image(image, masks)

def decode_coco_rle(rle, height, width):
    from pycocotools import mask as maskUtils
    return maskUtils.decode(rle).reshape((height, width))


def show_anns(anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    height, width = anns[0]['segmentation']['size']
    img = np.ones((height, width, 4))
    img[:, :, 3] = 0

    for ann in sorted_anns:
        rle = ann['segmentation']
        m = decode_coco_rle(rle, height, width)
        color_mask = np.concatenate([np.random.random(3), [0.9]])
        img[m == 1] = color_mask 
        if borders:
            import cv2
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1) 

    ax.imshow(img)
# %% Show image

def get_image_from_url(url):
  response = requests.get(url)
  return Image.open(BytesIO(response.content))

# image = Image.open('images/cars.jpg')
image = get_image_from_url("https://nichegamer.com/wp-content/uploads/2022/12/hogwarts-legacy-12-18-22-1.jpg")
# image = Image.open('images/truck.jpg')
plt.imshow(image)
plt.axis('off')
plt.show()

# %%
api = ClickingDatasetAPI()
response = api.get_masks(image,
min_mask_region_area=100.0,
pred_iou_thresh=0.94,
stability_score_thresh=0.7)
masks = response['masks']

# plot_bboxes_on_image(image, masks)

# Remove all masks with area less than threshold
# area_threshold = 400
# masks = [mask for mask in masks if mask['area'] >= area_threshold]

# %%
plt.figure(figsize=(20, 20))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.show() 

def crop_using_bbox(image, bbox, padding=10):
    x1, y1, width, height = map(int, bbox)
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)

    width += 2 * padding
    height += 2 * padding

    x2 = x1 + width
    y2 = y1 + height
    cropped_image = image.crop((x1, y1, x2, y2))
    return cropped_image

mask_count = len(masks)
n_cols = 2
num_rows = (mask_count + 3) // n_cols  # Calculate the number of rows needed for subplots
plt.figure(figsize=(20, 2 * num_rows))  

for i, mask in enumerate(masks):
    plt.subplot(num_rows, n_cols, i + 1)  # Set rows and n_cols columns

    # Decode RLE to binary mask
    binary_mask = decode_coco_rle(mask['segmentation'], image.height, image.width)

    # Crop bounding box around segmented pixels
    cropped_image = crop_using_bbox(image, mask['bbox'])

    plt.imshow(cropped_image)
    plt.title('Pixel area: ' + str(mask['area']) + ' iou pred: ' + str(mask['predicted_iou'])+ ' stability pred: ' + str(mask['stability_score']))
    plt.axis('off')
plt.show()

