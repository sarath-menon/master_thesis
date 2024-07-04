from clicker import Clicker
import torchvision
from torchvision import transforms

DATA_DIR = './datasets/label_studio_gen/coco_dataset/images'
ANNOTATIONS_FILE = './datasets/label_studio_gen/coco_dataset/result.json'

def load_dataset(data_dir, annFile):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    coco_dataset = torchvision.datasets.CocoDetection(root=data_dir, annFile=annFile, transform=transform)
    class_labels = [cat['name'] for cat in coco_dataset.coco.cats.values()]
    print(f"Dataset size: {len(coco_dataset)}")
    return coco_dataset, class_labels

def main():
    coco_dataset, class_labels = load_dataset(DATA_DIR, ANNOTATIONS_FILE)
    clicker = Clicker(class_labels)

    index = 1
    image, annotations = coco_dataset[index]
    click_points = clicker.get(image, annotations)
    clicker.show(image, annotations, click_point=click_points)


if __name__ == "__main__":
    main()

