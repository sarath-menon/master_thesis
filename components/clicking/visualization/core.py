
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw
import numpy as np
from clicking.visualization.bbox import BoundingBox, BBoxMode
from clicking.visualization.mask import SegmentationMask, SegmentationMode
import numpy as np
from pycocotools import mask as mask_utils
import cv2
import matplotlib.pyplot as plt


from dataclasses import dataclass

@dataclass
class BoundingBox:
    def __init__(self, x, y, w=None, h=None, x2=None, y2=None):
        if w is not None and h is not None:
            self.x1 = x
            self.y1 = y
            self.x2 = x + w
            self.y2 = y + h
        elif x2 is not None and y2 is not None:
            self.x1 = x
            self.y1 = y
            self.x2 = x2
            self.y2 = y2
        else:
            raise ValueError("Invalid parameters for bounding box.")


def show_localization_prediction(image, bboxes, labels):
    fig, ax = plt.subplots()

    # Display the image
    ax.imshow(image)

    # Plot each bounding box
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

# def show_masks( image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
#     for i, (mask, score) in enumerate(zip(masks, scores)):
#         plt.figure(figsize=(10, 10))
#         plt.imshow(image)
#         self.show_mask(mask, plt.gca(), borders=borders)
#         if point_coords is not None:
#             assert input_labels is not None
#             self.show_points(point_coords, input_labels, plt.gca())
#         if box_coords is not None:
#             self.show_box(box_coords, plt.gca())
            
#         if len(scores) > 1:
#             plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
#         plt.axis('off')
#         plt.show()

# def show_segmentation_prediction(image, masks, input_boxes, centroids):
#     plt.figure(figsize=(10, 10))
#     plt.imshow(image)
#     for mask, centroid in zip(masks, centroids):
#         if mask.shape[0] > 1:
#             mask = mask[0]

#         show_mask(mask.squeeze(), plt.gca(), random_color=True, centroid_point=centroid)
#     for box in input_boxes:
#         show_box(box, plt.gca())
#     plt.axis('off')
#     plt.show


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

def show_clickpoint(image, click_point, class_label):
    # Convert PIL Image to numpy array
    image_array = np.array(image)
    fig, ax = plt.subplots()
    
    ax.imshow(image_array)
    
    # Plot the click point
    ax.scatter(*click_point, marker='*', color='yellow', s=200, label=class_label)    
    ax.axis('off')
    plt.show()
    


# overlay bounding box in format (x, y, w, h) on a PIL image
def overlay_bounding_box(image, bbox: BoundingBox, color='red', thickness=10, padding=0):
    bbox = bbox.get(BBoxMode.XYXY)
    draw = ImageDraw.Draw(image)

    draw.rectangle((bbox[0] - padding, bbox[1] - padding, bbox[2] + padding, bbox[3] + padding), outline=color, width=thickness)
    return image
