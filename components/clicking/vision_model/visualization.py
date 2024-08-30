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
from clicking.vision_model.types import LocalizationResults, SegmentationResults
from typing import List, Dict
from clicking.common.pipeline_state import PipelineState


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

def show_clickpoint(image, click_point, object_name, size=100, color='yellow'):
    # Convert PIL Image to numpy array
    image_array = np.array(image)
    fig, ax = plt.subplots()
    
    ax.imshow(image_array)
    
    # Plot the click point
    ax.scatter(*click_point, marker='*', color=color, s=size, label=object_name)    
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

    for i, (object_name, response) in enumerate(responses.items()):
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
            ax.scatter(*centroid, marker='*', color=marker_color, s=marker_size, label=object_name)  

            # Plot class label as text in a box on top of the mask
            offset_y = 60
            ax.text(centroid[0], centroid[1] - offset_y, object_name, 
                    color=text_color, fontsize=text_size, 
                    bbox=dict(facecolor=textbox_color, edgecolor='none', alpha=1.0),
                    ha='center', va='center')

    ax.axis('off')
    plt.tight_layout()
    plt.show()


def show_localization_predictions(localization_results: LocalizationResults):
    if localization_results is None:
        print("No localization results available.")
        return

    for processed_sample in localization_results.processed_samples:
        image = processed_sample.image
        image_id = processed_sample.image_id
        predictions = localization_results.predictions[image_id]
        
        fig, ax = plt.subplots()
        ax.imshow(image)

        # Create a dictionary to map object names to unique IDs
        object_names = set(bbox.object_name for bbox in predictions)
        object_ids = {name: i for i, name in enumerate(object_names)}

        # Plot each bounding box
        for bbox in predictions:
            x, y, w, h = bbox.get(mode=BBoxMode.XYWH)
            bg_color = object_category_color_map(bbox.object_name)
            
            # Create a Rectangle patch
            rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor=bg_color, facecolor='none')
            ax.add_patch(rect)

            # Annotate the label
            object_id = object_ids[bbox.object_name]
            plt.text(x, y, str(object_id), color='white', fontsize=8, bbox=dict(facecolor=bg_color, alpha=0.9))

        # Print legend (id: object_name)
        for object_name, object_id in object_ids.items():
            print(f"{object_id}: {object_name}")

        # Remove the axis ticks and labels
        ax.axis('off')
        plt.show()

def object_category_color_map(object_name):
    # Create a dictionary to store color indices
    color_dict = {
        'Game Asset': (1, 0, 0),    # Red
        'Non-playable Character': (0, 1, 0),    # Green
        'Information Display': (0, 0, 1),    # Blue
    }

    # Default color if category is not found
    default_color = (0.5, 0.5, 0.5)  # Gray

    # Determine the category based on the object name
    if 'character' in object_name.lower():
        return color_dict['Non-playable Character']
    elif 'display' in object_name.lower() or 'ui' in object_name.lower():
        return color_dict['Information Display']
    else:
        return color_dict.get('Game Asset', default_color)


def show_segmentation_predictions(segmentation_results: SegmentationResults, textbox_color='red', text_color='white', text_size=12, marker_size=100, marker_color='yellow', mask_alpha=0.7, borders=False, label_offset_y=60):
    if segmentation_results is None:
        print("No segmentation results available.")
        return

    for processed_sample in segmentation_results.processed_samples:
        image = processed_sample.image
        image_id = processed_sample.image_id
        predictions = segmentation_results.predictions[image_id]

        # Create a new figure and axis
        fig, ax = plt.subplots(figsize=(10, 10))

        # Display the original image
        ax.imshow(image)

        total_classes = len(set(mask.object_name for mask in predictions))

        for i, mask in enumerate(predictions):
            m = mask_utils.decode(mask.get(SegmentationMode.COCO_RLE))
            color_mask = get_color(i, total_classes)

            # Create color overlay with correct shape and alpha channel
            color_overlay = np.zeros((*np.array(image).shape[:2], 4))
            color_overlay[m == 1] = [*color_mask, mask_alpha] 
            color_overlay[m == 0] = [0, 0, 0, 0]  
            ax.imshow(color_overlay)

            if borders:
                contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    ax.plot(contour[:, 0, 0], contour[:, 0, 1], color='white', linewidth=2)
            
            # Get mask centroid and plot it as click point
            centroid = get_mask_centroid(m)
            ax.scatter(*centroid, marker='*', color=marker_color, s=marker_size, label=mask.object_name)  

            # Plot class label as text in a box on top of the mask
            ax.text(centroid[0], centroid[1] - label_offset_y, mask.object_name, 
                    color=text_color, fontsize=text_size, 
                    bbox=dict(facecolor=textbox_color, edgecolor='none', alpha=1.0),
                    ha='center', va='center')

        ax.axis('off')
        plt.title(f"Image ID: {image_id}")
        plt.tight_layout()
        plt.show()

        # Print legend (object_name: description)
        print(f"Legend for Image ID: {image_id}")
        for mask in predictions:
            print(f"{mask.object_name}: {mask.description}")
        print("\n")
