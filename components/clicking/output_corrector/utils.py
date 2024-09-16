from PIL import ImageDraw
from clicking.common.bbox import BoundingBox, BBoxMode

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