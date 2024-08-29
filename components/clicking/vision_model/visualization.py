
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw
import numpy as np
from clicking.vision_model.bbox import BoundingBox, BBoxMode
from clicking.vision_model.mask import SegmentationMask, SegmentationMode
import numpy as np
from pycocotools import mask as mask_utils
import cv2
import matplotlib.pyplot as plt
from typing import List
from clicking.vision_model.types import SegmentationResp, LocalizationResp
from clicking.vision_model.utils import get_mask_centroid

from dataclasses import dataclass

def show_mask( mask, ax, random_color=False, borders=True, centroid_point=None):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask = mask.astype(np.uint8)
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

        if centroid_point is not None:
            ax.scatter(centroid_point[0], centroid_point[1], color='red', marker='*', s=100)

        if borders:
            import cv2
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)
        ax.imshow(mask_image)

def show_points( coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box( box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))



def show_segmentation_prediction(image, masks):
    # Assuming 'image' is your original PIL Image
    image_array = np.array(image)

    # Create a new figure and axis
    fig, ax = plt.subplots(figsize=(10, 10))

    # Display the original image
    ax.imshow(image)

    borders = False
    mask_alpha = 0.7

    for mask in masks:
        m = mask_utils.decode(mask.get(SegmentationMode.COCO_RLE))
        color_mask = np.random.random(3)

        # Create color overlay with correct shape and alpha channel
        color_overlay = np.zeros((*image_array.shape[:2], 4))
        color_overlay[m == 1] = [*color_mask, mask_alpha] 
        color_overlay[m == 0] = [0, 0, 0, 0]  
        ax.imshow(color_overlay)

        if borders:
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                ax.plot(contour[:, 0, 0], contour[:, 0, 1], color='white', linewidth=2)

    ax.axis('off')
    plt.tight_layout()
    plt.show()

def show_clickpoint(image, click_point, class_label, size=100, color='yellow'):
    # Convert PIL Image to numpy array
    image_array = np.array(image)
    fig, ax = plt.subplots()
    
    ax.imshow(image_array)
    
    # Plot the click point
    ax.scatter(*click_point, marker='*', color=color, s=size, label=class_label)    
    ax.axis('off')
    plt.show()

# overlay bounding box in format (x, y, w, h) on a PIL image
def overlay_bounding_box(image, bbox: BoundingBox, color='red', thickness=10, padding=0):
    bbox = bbox.get(BBoxMode.XYXY)
    draw = ImageDraw.Draw(image)

    draw.rectangle((bbox[0] - padding, bbox[1] - padding, bbox[2] + padding, bbox[3] + padding), outline=color, width=thickness)
    return image

def get_color(index, total):
    # Create a dictionary to store color indices
    color_dict = {
        0: (1, 0, 0),    # Red
        1: (0, 1, 0),    # Green
        2: (0, 0, 1),    # Blue
        3: (1, 1, 0),    # Yellow
        4: (1, 0, 1),    # Magenta
        5: (0, 1, 1),    # Cyan
    }
    
    # Use modulo to cycle through colors if index exceeds dictionary size
    color_index = index % len(color_dict)
    
    return color_dict[color_index]

def show_clickpoint_predictions(image, responses: List[SegmentationResp], textbox_color='red', text_color='white', text_size=12, marker_size=100, marker_color='yellow'):
    # Assuming 'image' is your original PIL Image
    image_array = np.array(image)

    # Create a new figure and axis
    fig, ax = plt.subplots(figsize=(10, 10))

    # Display the original image
    ax.imshow(image)

    borders = False
    mask_alpha = 0.7
    total_classes = len(responses)

    for i, (class_label, response) in enumerate(responses.items()):
        masks = [SegmentationMask(mask, mode=SegmentationMode.COCO_RLE) for mask in response.prediction.masks]
        for mask in masks:
            m = mask_utils.decode(mask.get(SegmentationMode.COCO_RLE))
            color_mask = get_color(i, total_classes)

            # Create color overlay with correct shape and alpha channel
            color_overlay = np.zeros((*image_array.shape[:2], 4))
            color_overlay[m == 1] = [*color_mask, mask_alpha] 
            color_overlay[m == 0] = [0, 0, 0, 0]  
            ax.imshow(color_overlay)

            if borders:
                contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    ax.plot(contour[:, 0, 0], contour[:, 0, 1], color='white', linewidth=2)
            
            # get mask centroid and plot it as click point
            centroid = get_mask_centroid(m)
            ax.scatter(*centroid, marker='*', color=marker_color, s=marker_size, label=class_label)  

            # Plot class label as text in a box on top of the mask
            offset_y = 60
            ax.text(centroid[0], centroid[1] - offset_y, class_label, 
                    color=text_color, fontsize=text_size, 
                    bbox=dict(facecolor=textbox_color, edgecolor='none', alpha=1.0),
                    ha='center', va='center')

    ax.axis('off')
    plt.tight_layout()
    plt.show()

def show_localization_predictions(image, responses: List[LocalizationResp]):
    fig, ax = plt.subplots()

    # Display the image
    ax.imshow(image)

    # Plot each bounding box
    for response in responses:
        bboxes = response.prediction.bboxes
        labels = response.prediction.labels
        for bbox, label in zip(bboxes, labels):
            # Unpack the bounding box coordinates
            x1, y1, x2, y2 = bbox
            # Create a Rectangle patch
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')
            # Add the rectangle to the Axes
            ax.add_patch(rect)
            # Annotate the label
            plt.text(x1, y1, label, color='white', fontsize=8, bbox=dict(facecolor='red', alpha=0.5))

    # Remove the axis ticks and labels
    ax.axis('off')
    
    # Show the plot
    plt.show()


from clicking.vision_model.types import LocalizationResp
from typing import List, Dict

def show_localization_predictions(image, responses: List[LocalizationResp], categories: Dict[str, str], descriptions: Dict[str, str], text_color='white'):
    fig, ax = plt.subplots()

    ax.imshow(image)

    # Enumerate the descriptions dict 
    description_ids = {description: i for i, description in enumerate(set(descriptions))}

    # Plot each bounding box
    for (object_name, response) in responses.items():
        bboxes = response.prediction.bboxes
        category = categories[object_name]

        for (i, bbox) in enumerate(bboxes):
            # Unpack the bounding box coordinates
            x1, y1, x2, y2 = bbox
            bg_color = object_category_color_map(category)
            
            # Create a Rectangle patch
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor=bg_color, facecolor='none')
            
            # Add the rectangle to the Axes
            ax.add_patch(rect)

            # Annotate the label
            description_id = description_ids[object_name]
            plt.text(x1, y1, str(description_id), color=text_color, fontsize=8, bbox=dict(facecolor=bg_color, alpha=0.9))

    # legend (object_name: description)
    # object_names = list(responses.keys())
    # for object_name in object_names:
    #     print(f"{object_name}: {descriptions[object_name]}")

    # legend (id: object_name)
    for id, object_name in description_ids.items():
        print(f"{id}: {object_name}")

    # Remove the axis ticks and labels
    ax.axis('off')
    plt.show()

def object_category_color_map(category):
    # Create a dictionary to store color indices
    color_dict = {
        'Game Asset': (1, 0, 0),    # Red
        'Non-playable Character': (0, 1, 0),    # Green
        'Information Display': (0, 0, 1),    # Blue
    }

    return color_dict[category]