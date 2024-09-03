
from clicking.common.types import ClickingImage

from prettytable import PrettyTable
from typing import List 
from matplotlib import pyplot as plt

def print_image_objects(image_objects: List[ClickingImage], show_image=False):
    for result in image_objects:
        
        predicted_objects = [[i, obj.name, obj.category.value if obj.category else 'No Category'] for i, obj in enumerate(result.predicted_objects)]

        annotated_objects = [[i, obj.name] for i, obj in enumerate(result.annotated_objects)]
        
        # Pad the shorter list to match the length of the longer list
        max_length = max(len(predicted_objects), len(annotated_objects))
        predicted_objects += [['', '', '']] * (max_length - len(predicted_objects))
        annotated_objects += [['', '']] * (max_length - len(annotated_objects))
        
        # Combine predicted and true objects
        combined_objects = [[i if i < len(predicted_objects) else '', p[1], p[2], t[1]] for i, (p, t) in enumerate(zip(predicted_objects, annotated_objects))]
        
        table = PrettyTable()
        table.field_names = ["Index", "Predicted object", "Category", "Annotated objects"]
        table.add_rows(combined_objects)
        
        print(f"Image ID: {result.id}")

        if show_image:
            plt.imshow(result.image)
            plt.axis('off')
            
        plt.show()
        print(table)
        print("\n")

def print_object_descriptions(image_objects: List[ClickingImage], show_image=False):
    for result in image_objects:
        table = PrettyTable()
        table.field_names = ["Index", "Predicted Object", "Description"]
        
        for i, obj in enumerate(result.predicted_objects):
            table.add_row([i, obj.name, obj.description or "No description available"])
        
        print(f"Image ID: {result.id}")

        # show image
        if show_image:
            plt.imshow(result.image)
            plt.axis('off')

        plt.show()
        print(table)
        print("\n")

def selva():
    print("Selva")