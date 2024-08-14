from models.sam2 import SAM2Model
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

sam2 = SAM2Model()
image = Image.open('images/truck.jpg')
# masks, scores = sam2.predict_with_bbox(image, np.array([425, 600, 700, 875]))

#%% batch bbox
input_boxes = np.array([
    [75, 275, 1725, 850],
    [425, 600, 700, 875],
    [1375, 550, 1650, 800],
    [1240, 675, 1400, 750],
])
masks_batch, scores_batch = sam2.batched_prediction(image, input_boxes)
print(masks_batch)

# plt.figure(figsize=(10, 10))
# plt.imshow(image)
# for mask in masks:
#     self.show_mask(mask.squeeze(0), plt.gca(), random_color=True)
# for box in input_boxes:
#     self.show_box(box, plt.gca())
# plt.axis('off')
# plt.show()