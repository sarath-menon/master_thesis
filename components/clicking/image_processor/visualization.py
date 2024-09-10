import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
from clicking.common.bbox import BoundingBox, BBoxMode
from clicking.common.mask import SegmentationMask, SegmentationMode
from pycocotools import mask as mask_utils
import cv2
from typing import List
from clicking.common.data_structures import ClickingImage, ImageObject, ObjectCategory, CATEGORY_COLOR_MAP
from PIL import ImageDraw
from clicking.common.logging import print_object_descriptions
from clicking.common.data_structures import ValidityStatus

# overlay bounding box in format (x, y, w, h) on a PIL image
def overlay_bounding_box(image, bbox: BoundingBox, color='red', thickness=14, padding=0):
    bbox = bbox.get(BBoxMode.XYXY)
    draw = ImageDraw.Draw(image)

    # Ensure the bounding box is within the image boundaries
    width, height = image.size
    x1 = max(0, min(bbox[0] - padding, width - 1))
    y1 = max(0, min(bbox[1] - padding, height - 1))
    x2 = max(0, min(bbox[2] + padding, width - 1))
    y2 = max(0, min(bbox[3] + padding, height - 1))

    # Adjust thickness if the box is too small
    box_width = x2 - x1
    box_height = y2 - y1
    thickness = min(thickness, min(box_width, box_height) // 2)

    draw.rectangle((x1, y1, x2, y2), outline=color, width=thickness)
    return image

def get_color(index):
    color_dict = {
        0: 'red',    
        1: 'green',   
        2: 'blue',    
        3: 'yellow',    # Yellow
        4: 'magenta',
        5: 'cyan',
    }
    color_index = index % len(color_dict)
    return color_dict[color_index]


def show_clickpoint_predictions(clicking_image: ClickingImage, textbox_color='red', text_color='white', text_size=12, marker_size=100, marker_color='yellow', object_names_to_show=None):
    image_array = np.array(clicking_image.image)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(clicking_image.image)

    mask_alpha = 0.7
    total_objects = len(clicking_image.predicted_objects)

    for i, obj in enumerate(clicking_image.predicted_objects):

        if object_names_to_show and obj.name not in object_names_to_show:
            continue

        m = mask_utils.decode(obj.mask.get(SegmentationMode.COCO_RLE))

        color_mask = get_color(i)
        # Convert color_mask to RGB values
        rgb_color = plt.cm.colors.to_rgb(color_mask)
        
        color_overlay = np.zeros((*np.array(clicking_image.image).shape[:2], 4), dtype=np.float32)
        m_expanded = np.expand_dims(m, axis=-1)
        color_overlay[m_expanded[:, :, 0] == 1] = [*rgb_color, mask_alpha]

        ax.imshow(color_overlay)

        contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            ax.plot(contour[:, 0, 0], contour[:, 0, 1], color='white', linewidth=2)
        
        centroid = get_mask_centroid(m)
        ax.scatter(*centroid, marker='*', color=marker_color, s=marker_size, label=obj.name)  

        offset_y = 60
        ax.text(centroid[0], centroid[1] - offset_y, obj.name, 
                color=text_color, fontsize=text_size, 
                bbox=dict(facecolor=textbox_color, edgecolor='none', alpha=0.7),
                ha='center', va='center')

    ax.axis('off')
    plt.title(f"Image ID: {clicking_image.id}")
    plt.tight_layout()
    plt.show()

def show_localization_predictions(clicking_image: ClickingImage, object_names_to_show=None, show_descriptions=True, label_alpha=0.7, label_y_offset = 30):
    fig, ax = plt.subplots()
    ax.imshow(clicking_image.image)

    object_names = set(obj.name for obj in clicking_image.predicted_objects)
    object_ids = {name: i for i, name in enumerate(object_names)}

    for i, obj in enumerate(clicking_image.predicted_objects):
        if obj.bbox is None:
            print(f"Object {obj.name} has no bounding box")
            continue

        if object_names_to_show is not None and obj.name not in object_names_to_show:
            print(f"Object {obj.name} not in object_names_to_show")
            continue

        x, y, w, h = obj.bbox.get(mode=BBoxMode.XYWH)

        #default to gray if category is not found
        bbox_color = CATEGORY_COLOR_MAP.get(obj.category, 'gray') 
        bg_color = 'red'

        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=bbox_color, facecolor='none')
        ax.add_patch(rect)

        object_id = object_ids[obj.name]
        plt.text(x , y - label_y_offset, f"{object_id}:{obj.name}", color='white', fontsize=8, bbox=dict(facecolor=bg_color, alpha=label_alpha))

    ax.axis('off')
    plt.show()

    if show_descriptions:
        print_object_descriptions([clicking_image], show_image=False)
        print("\n") 

def show_segmentation_predictions(clicking_image: ClickingImage, textbox_color='red', text_color='white', text_size=12, marker_size=100, marker_color='yellow', mask_alpha=0.7, label_offset_y=60, object_names_to_show=None,show_descriptions=False):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(clicking_image.image)

    for i, obj in enumerate(clicking_image.predicted_objects):
        if obj.mask is None:
            print(f"Warning: Object '{obj.name}' has no segmentation mask.")
            continue

        if obj.validity.status is ValidityStatus.INVALID:
            print(f"Skipping segmentation for {obj.name} because it is invalid: {obj.validity.status}")
            continue

        if object_names_to_show and obj.name not in object_names_to_show:
            continue

        m = mask_utils.decode(obj.mask.get(SegmentationMode.COCO_RLE))

        color_mask = get_color(i)
        # Convert color_mask to RGB values
        rgb_color = plt.cm.colors.to_rgb(color_mask)
        
        color_overlay = np.zeros((*np.array(clicking_image.image).shape[:2], 4), dtype=np.float32)
        m_expanded = np.expand_dims(m, axis=-1)
        color_overlay[m_expanded[:, :, 0] == 1] = [*rgb_color, mask_alpha]

        ax.imshow(color_overlay)

        contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            ax.plot(contour[:, 0, 0], contour[:, 0, 1], color='white', linewidth=2)
        
        centroid = get_mask_centroid(m)
        if centroid is None:
            print(f"Warning: Object '{obj.name}' has no centroid")
            continue

        ax.scatter(*centroid, marker='*', color=marker_color, s=marker_size, label=obj.name)  

        ax.text(centroid[0], centroid[1] - label_offset_y, obj.name, 
                color=text_color, fontsize=text_size, 
                bbox=dict(facecolor=textbox_color, edgecolor='none', alpha=0.7),
                ha='center', va='center')

    ax.axis('off')
    plt.title(f"Image ID: {clicking_image.id}")
    plt.tight_layout()
    plt.show()

    if show_descriptions:
        print(f"Legend for Image ID: {clicking_image.id}")
        print_object_descriptions([clicking_image], show_image=False)
        print("\n") 

def get_mask_centroid(mask):
    moments = cv2.moments(mask.astype(np.uint8))
    if moments["m00"] != 0:
        centroid_x = int(moments["m10"] / moments["m00"])
        centroid_y = int(moments["m01"] / moments["m00"])
        return centroid_x, centroid_y
    else:
        return None