import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
from clicking.common.bbox import BoundingBox, BBoxMode
from clicking.common.mask import SegmentationMask, SegmentationMode
from pycocotools import mask as mask_utils
import cv2
from typing import List
from clicking.common.data_structures import *
from PIL import ImageDraw
from clicking.common.logging import print_object_descriptions
from clicking.common.data_structures import ValidityStatus
plt.style.use('dark_background')
from collections import defaultdict, Counter

from prettytable import PrettyTable
from ..output_corrector.core import VerificationMode

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

import random

colormap = ['blue','orange','green','purple','brown','pink','gray','olive','cyan','red',
            'lime','indigo','violet','aqua','magenta','coral','gold','tan','skyblue']
            
def show_ocr_boxes(clicking_image: ClickingImage, prediction, scale=1):
    image = clicking_image.image.copy()
    draw = ImageDraw.Draw(image)
    quad_boxes, labels = prediction.bboxes, prediction.labels

    plt.figure(figsize=(12, 12))
    plt.imshow(image)
    plt.grid(False)
    plt.axis('off')

    for box, label in zip(quad_boxes, labels):
        color = random.choice(colormap)
        new_box = (np.array(box) * scale).tolist()
        draw.polygon(new_box, width=3, outline=color)
        draw.text((new_box[0]+8, new_box[1]+2),
                  f"{label}",
                  align="left",
                  fill=color)

    plt.imshow(image)
    plt.show()


def show_invalid_objects(state: PipelineState, mode: VerificationMode):
    for image in state.images:
        for obj in image.predicted_objects:
            if obj.validity.status != ValidityStatus.INVALID:
                continue

            print(f"Object: {obj.name} is invalid because: {obj.validity.reason}")

            if mode == VerificationMode.CROP_BBOX:
                if obj.bbox is None:
                    print(f"Object {obj.name} has no bounding box")
                    continue
                plt.imshow(obj.bbox.extract_area(image.image, padding=10))
            elif mode == VerificationMode.CROP_MASK:
                plt.imshow(obj.mask.extract_area(image.image, padding=10))
            else:
                raise ValueError(f"Invalid verification mode: {mode}")
            
            plt.axis('off')
            plt.show()


def compare_invalid_objects(states: List[PipelineState], labels: List[str], visualize=False, show_details=False):
    invalid_objects = defaultdict(lambda: defaultdict(bool))
    
    for state, label in zip(states, labels):
        for image in state.images:
            for obj in image.predicted_objects:
                if obj.validity.status == ValidityStatus.INVALID:
                    invalid_objects[obj.name][label] = True
    
    statistics = {obj_name: sum(invalid_in_states.values()) for obj_name, invalid_in_states in invalid_objects.items()}
    sorted_stats = dict(sorted(statistics.items(), key=lambda x: x[1], reverse=True))
    
    if visualize:
        visualize_invalid_objects(sorted_stats, len(states))
    
    if show_details:
        print_detailed_information(sorted_stats, invalid_objects, len(states))
    
    return sorted_stats, invalid_objects

def visualize_invalid_objects(invalid_object_stats, num_states):
    invalid_count_frequency = Counter(invalid_object_stats.values())
    invalid_counts = list(range(1, num_states + 1))
    frequencies = [invalid_count_frequency.get(count, 0) for count in invalid_counts]

    plt.figure(figsize=(10, 6))
    plt.bar(invalid_counts, frequencies, align='center', alpha=0.8)
    plt.xlabel('Number of States Where Object is Invalid')
    plt.ylabel('Number of Objects')
    plt.title('Distribution of Invalid Objects Across States')
    plt.xticks(invalid_counts)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for i, v in enumerate(frequencies):
        if v > 0:
            plt.text(i + 1, v, str(v), ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

def print_detailed_information(invalid_object_stats, invalid_objects, num_states):
    print("\nDetailed invalid object information:")
    table = PrettyTable()
    table.field_names = ["Object Name", "Invalid Count", "Invalid States"]
    table.align["Object Name"] = "l"
    table.align["Invalid Count"] = "center"
    table.align["Invalid States"] = "l"

    for obj_name, invalid_count in invalid_object_stats.items():
        invalid_in = [label for label, is_invalid in invalid_objects[obj_name].items() if is_invalid]
        table.add_row([obj_name, f"{invalid_count}/{num_states}", ", ".join(invalid_in)])

    print(table)


def show_invalid_object_images(states: List[PipelineState], state_labels: List[str], sorted_stats: Dict[str, int], max_images_per_object: int = 3):
    """
    Display images containing invalid objects for each category.

    Args:
    sorted_stats (Dict[str, int]): A dictionary of object names and their invalid counts, sorted in descending order.
    states (List[PipelineState]): List of PipelineState objects containing the images and object information.
    max_images_per_object (int): Maximum number of images to display for each object category.
    """
    sorted_objects = sorted(sorted_stats.items(), key=lambda x: x[1], reverse=True)

    for obj_name, invalid_count in sorted_objects:
        print("\n" + "=" * 50)
        print(f"Object: {obj_name}")
        print(f"Invalid in {invalid_count} out of {len(states)} states")
        print("=" * 50)
        
        images_shown = 0
        for state, label in zip(states, state_labels):
            for image in state.images:
                for obj in image.predicted_objects:
                    if obj.name != obj_name or obj.validity.status != ValidityStatus.INVALID:
                        continue

                    print(f"State: {label}")
                    print(f"Reason: {obj.validity.reason}")
                    plt.figure(figsize=(10, 10))
                    plt.imshow(image.image)
                    plt.title(f"{obj_name} - Image ID: {image.id}")
                    plt.axis('off')
                    
                    if obj.bbox:
                        bbox = obj.bbox.get(BBoxMode.XYXY)
                        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
                                                 linewidth=2, edgecolor='r', facecolor='none')
                        plt.gca().add_patch(rect)
                    
                    if obj.mask:
                        mask = mask_utils.decode(obj.mask.get(SegmentationMode.COCO_RLE))
                        plt.imshow(mask, alpha=0.5, cmap='jet')
                    
                    plt.show()

                    
                    images_shown += 1
                    if images_shown >= max_images_per_object:
                        break
                
                if images_shown >= max_images_per_object:
                    break
            
            if images_shown >= max_images_per_object:
                break


def show_ui_elements(clicking_image: ClickingImage, label_alpha=0.7, label_y_offset=30, scale=1, bbox_thickness=5):
    fig, ax = plt.subplots()
    ax.imshow(clicking_image.image)

    draw = ImageDraw.Draw(clicking_image.image)

    buttons = []
    
    for ui_element in clicking_image.ui_elements:

        if ui_element.name is None:
            continue

        if ui_element.bbox is None:
            print(f"UI Element {ui_element.name} has no bounding box")
            continue
            
        color = random.choice(colormap)
        new_box = (np.array(ui_element.bbox) * scale).tolist()
        draw.polygon(new_box, width=bbox_thickness, outline=color)
        draw.text((new_box[0]+8, new_box[1]+2),
                  f"{ui_element.name}",
                  align="left",
                  fill=color)

        if ui_element.category == "Button":
            buttons.append(ui_element.name)

        # x, y, w, h = ui_element.bbox.get(mode=BBoxMode.XYWH)

        # bbox_color = 'blue'  # You can adjust this color as needed
        # bg_color = 'red'

        # rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=bbox_color, facecolor='none')
        # ax.add_patch(rect)

        # plt.text(x, y - label_y_offset, f"{ui_element.name}", color='white', fontsize=8, bbox=dict(facecolor=bg_color, alpha=label_alpha))

    ax.axis('off')
    plt.show()

    print(f"Button: {buttons}")