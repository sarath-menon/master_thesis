
from clicking.common.data_structures import ClickingImage
from clicking.common.data_structures import PipelineState
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

def print_object_descriptions(image_objects: List[ClickingImage], show_image=False, max_col_width=30, show_stats=False):
    category_counts = {}

    for result in image_objects:
        table = PrettyTable()
        table.field_names = ["Index", "Predicted Object", "Description", "Reasoning"]
        # Set the maximum width for each column
        table.max_width = max_col_width
        
        for i, obj in enumerate(result.predicted_objects):
            table.add_row([i, f"{obj.name}\n({obj.category.value})", obj.description or "No description available", obj.significance or "No reasoning available"])

            if show_stats:
                category = obj.category.value if obj.category else 'No Category'
                category_counts[category] = category_counts.get(category, 0) + 1
        
        print(f"Image ID: {result.id}")

        # show image
        if show_image:
            plt.imshow(result.image)
            plt.axis('off')                    

        plt.show()
        print(table)
        print("\n")

    if show_stats:   
        plt.figure(figsize=(12, 6))
        font_size = 16
        plt.style.use('dark_background')
        bars = plt.bar(category_counts.keys(), category_counts.values())
        plt.title('Histogram of Object Categories Across All Images', color='white', fontsize=font_size)
        plt.xticks(fontsize=font_size)
        plt.yticks(fontsize=font_size)

        # Add value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height}',
                        ha='center', va='bottom', color='white', fontsize=font_size)

        plt.tight_layout()


def show_object_validity(result: PipelineState):
    table = PrettyTable()
    table.field_names = ["Object Name", "Is Valid", "Accuracy", "Visibility", "Reason"]
    table.align = "l"  # Align all columns to the left
    
    for image in result.images:
        for obj in image.predicted_objects:
            table.add_row([
                obj.name,
                obj.validity.status,
                obj.validity.accuracy,
                obj.validity.visibility,
                obj.validity.reason
            ])
    
    print(table)

def print_ocr_results(result: PipelineState):
    table = PrettyTable()
    table.field_names = ["Text"]
    table.align = "l"  # Align all columns to the left

    for label in result.prediction.labels:
        table.add_row([
            label
        ])
    
    print(table)