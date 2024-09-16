from clicking.image_processor.visualization import overlay_bounding_box
from clicking.common.data_structures import PipelineState
from typing import List, Dict
import os
import json
from pydantic import BaseModel, Field
from evaluate import load
from collections import Counter
from typing import List, Dict, Literal
from clicking.common.data_structures import *
import wandb
import pandas as pd
from collections import Counter
from clicking.pipeline.core import PipelineState
import matplotlib.pyplot as plt
from collections import defaultdict

PROJECT_NAME = "clicking"

class ChoiceResult(BaseModel):
    value: Dict[str, List[str]]
    from_name: str
    to_name: str
    type: Literal["choices"]

class Prediction(BaseModel):
    model_name: str = ""
    result: List[ChoiceResult]

class FormattedData(BaseModel):
    data: Dict[str, str]
    annotations: List[dict] = []
    predictions: List[Prediction] = []

def save_validity_results(results: PipelineState, output_folder: str):
    images_folder = os.path.join(output_folder, 'images')
    json_file = os.path.join(output_folder, 'validity_results.json')

    if not os.path.exists(images_folder):
        os.makedirs(images_folder)
    
    entries = []
    for clicking_image in results.images:
        image = clicking_image.image

        for obj in clicking_image.predicted_objects:
            overlay_image = overlay_bounding_box(image.copy(), obj.bbox, thickness=10)

            filename = f"{clicking_image.id}_{obj.name}.jpg"
            overlay_image.save(os.path.join(images_folder, filename))

            validity_choice = "valid" if obj.validity.status == ValidityStatus.VALID else "invalid"

            choice_result = ChoiceResult(
                value={"choices": [validity_choice]},
                from_name="Labelling",
                to_name="image",
                type="choices"
            )

            formatted_data = FormattedData(
                data={
                    "image": f"/data/local-files/?d=evals/output_corrector/images/{clicking_image.id}_{obj.name}.jpg",
                    "label": obj.name,
                    "description": obj.description
                },
                predictions=[Prediction(result=[choice_result])]
            )
            entries.append(formatted_data.dict())

    with open(json_file, 'w') as f:
        json.dump(entries, f, indent=2)
    
    print(f"ObjectValidity results saved to {output_folder}")
    print(f"Overlay images saved in the same directory")

def evaluate_validity_results(ground_truth_file: str, predictions_file: str):
    # Load the metric
    accuracy_metric = load("accuracy")
    
    # Load ground truth data
    with open(ground_truth_file, 'r') as f:
        ground_truth = json.load(f)
    
    # Load predictions
    with open(predictions_file, 'r') as f:
        predictions = json.load(f)
    
    # Prepare data for evaluation
    gt_labels = []
    pred_labels = []
    
    for gt_item in ground_truth:
        gt_image_id = gt_item['meta']['image_id']
        gt_label = gt_item['annotations'][0]['result'][1]['value']['choices'][0]  # 'correct' or 'incorrect'
        
        # Find corresponding prediction
        pred_item = next((p for p in predictions if p['image_id'] == gt_image_id), None)
        
        if pred_item:
            # Convert string labels to integers
            gt_labels.append(1 if gt_label == 'correct' else 0)
            pred_labels.append(1 if pred_item['is_valid'] else 0)
    
    # Calculate accuracy
    results = accuracy_metric.compute(references=gt_labels, predictions=pred_labels)
    
    # Count correct and incorrect predictions
    correct_count = sum(1 for gt, pred in zip(gt_labels, pred_labels) if gt == pred)
    incorrect_count = len(gt_labels) - correct_count
    
    # Count predictions by class
    gt_counter = Counter(gt_labels)
    pred_counter = Counter(pred_labels)
    
    # Add counts to results
    results['total_predictions'] = len(gt_labels)
    results['correct_predictions'] = correct_count
    results['incorrect_predictions'] = incorrect_count
    results['ground_truth_counts'] = dict(gt_counter)
    results['prediction_counts'] = dict(pred_counter)
    
    # Print results
    print(f"Accuracy: {results['accuracy']:.2f}")
    print(f"Correct predictions: {results['correct_predictions']} / {results['total_predictions']} ({results['accuracy']:.2%})")
    return results

def save_image_descriptions(clicking_images: List[ClickingImage], output_folder: str, prompt_path: str):
    descriptions = []
    for image in clicking_images:
        for obj in image.predicted_objects:
            descriptions.append({
                "image_id": image.id,
                "object_id": str(obj.id),
                "name": obj.name,
                "category": obj.category.value,
                "description": obj.description
            })
    
    output_file = os.path.join(output_folder, 'image_descriptions.json')
    with open(output_file, 'w') as f:
        json.dump(descriptions, f, indent=2)

    # Convert descriptions to a pandas DataFrame
    descriptions_df = pd.DataFrame(descriptions)

    # # Log DataFrame to file as a markdown table
    # output_markdown_file = os.path.join(output_folder, 'image_descriptions.md')
    # with open(output_markdown_file, 'w') as f:
    #     f.write(descriptions_df.to_markdown(index=False))
    
    # print(f"Image descriptions saved as markdown table to {output_markdown_file}")

    # # Log data to W&B as a table
    # run = wandb.init(project=PROJECT_NAME)

    # table = wandb.Table(dataframe=descriptions_df)
    # wandb.log({"image_descriptions": table})
    
    # # save content of the prompt to wandb
    # with open(prompt_path, 'r') as f:
    #     prompt_content = f.read()
    # wandb.log({"prompt": prompt_content})
    # run.notes = prompt_content

    # artifact = wandb.Artifact(name = "example_artifact", type = "dataset")
    # artifact.add_file(local_path=prompt_path, name="prompt")
    # run.log_artifact(artifact)

    # run.finish()

   
    # # Log data to MLflow
    # remote_server_uri = "http://127.0.0.1:8084"  # set to your server URI
    # mlflow.set_tracking_uri(remote_server_uri)
    # mlflow.set_experiment("/my-experiment")

    # with mlflow.start_run():
    #     mlflow.log_table(descriptions_df, "./selva.json")

    # print(f"Image descriptions saved to {output_file}")

def analyze_overall_validity(states: List[PipelineState]) -> PipelineState:
    image_validity = defaultdict(lambda: ObjectValidity(status=ValidityStatus.INVALID))
    
    for state in states:
        for image in state.images:
            current_validity = image_validity[image.id]
            
            # Check if any object in the image is valid
            if any(obj.validity.status == ValidityStatus.VALID for obj in image.predicted_objects):
                current_validity.status = ValidityStatus.VALID
    
    # Create a new PipelineState with overall validity information
    overall_state = PipelineState()
    
    for state in states:
        for image in state.images:
            overall_validity = image_validity[image.id]
            
            # Create a new ClickingImage with an ImageObject representing overall validity
            overall_image = ClickingImage(
                image=image.image,
                id=image.id,
                predicted_objects=[
                    ImageObject(
                        name="Overall Validity",
                        validity=overall_validity
                    )
                ]
            )
            
            overall_state.images.append(overall_image)
    
    # Calculate and print statistics
    total_images = len(overall_state.images)
    valid_images = sum(1 for img in overall_state.images if img.predicted_objects[0].validity.status == ValidityStatus.VALID)
    invalid_images = total_images - valid_images
    
    print(f"Valid Images: {valid_images}/{total_images} ({valid_images/total_images:.2%})")
    
    return overall_state

def show_validity_statistics(states: List[PipelineState], labels: List[str]):
    plt.figure(figsize=(15, 8))

    # calculate overall validity
    overall_validity_results = analyze_overall_validity(states)
    states.append(overall_validity_results)
    labels.append("Overall Validity")
    
    statuses = list(ValidityStatus)
    # remove UNKNOWN from statuses
    statuses = [status for status in statuses if status != ValidityStatus.UNKNOWN]

    width = 0.2
    x = np.arange(len(statuses))
    
    colors = ['red', 'green', 'gray', 'yellow']  # More distinct colors
    
    for i, (state, label) in enumerate(zip(states, labels)):
        validity_counts = Counter()
        
        for image in state.images:
            for obj in image.predicted_objects:
                validity_counts[obj.validity.status] += 1
        
        total_objects = sum(validity_counts.values())
        percentages = [validity_counts[status] / total_objects * 100 if total_objects > 0 else 0 for status in statuses]
        
        offset = width * (i - 0.5 * (len(states) - 1))
        bars = plt.bar(x + offset, percentages, width, label=label, color=colors[i % len(colors)], edgecolor='black', linewidth=1.5)
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height,
                     f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.xlabel('Validity Status', fontsize=12)
    plt.ylabel('Percentage', fontsize=12)
    plt.title('Object Validity Statistics Comparison', fontsize=16)
    plt.legend(fontsize=12, loc='upper left', bbox_to_anchor=(1, 1))

    plt.ylim(0, 100)
    
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()
    from tabulate import tabulate
    table_data = []
    headers = ["Label", "Total Objects"] + [status.name for status in statuses]

    for index, (state, label) in enumerate(zip(states, labels), start=1):
        validity_counts = Counter()
        for image in state.images:
            for obj in image.predicted_objects:
                validity_counts[obj.validity.status] += 1
        
        total_objects = sum(validity_counts.values())
        row = [index, label, total_objects]
        for status in statuses:
            count = validity_counts[status]
            percentage = (count / total_objects) * 100 if total_objects > 0 else 0
            row.append(f"{count} ({percentage:.2f}%)")
        table_data.append(row)

    max_column_width = 15
    headers = ["Index"] + headers
    print(tabulate(table_data, headers=headers, tablefmt="pretty", 
                   maxcolwidths=[5, 15, 5] + [max_column_width] * len(statuses),
                   numalign="center", stralign="center"))
    print()  # Add an extra newline for spacing between rows


def plot_ui_element_histogram(images: List[ClickingImage]):
    with_bbox = 0
    without_bbox = 0

    for image in images:
        for element in image.ui_elements:
            if element.category == "Button" and element.name is not None:
                if element.bbox is not None:
                    with_bbox += 1
                else:
                    without_bbox += 1

    total = with_bbox + without_bbox
    categories = ['With BBox', 'Without BBox']
    percentages = [with_bbox / total * 100, without_bbox / total * 100]
 
    plt.figure(figsize=(10, 6))
    plt.bar(categories, percentages)
    plt.title('UI Elements (Buttons)')
    plt.ylabel('Percentage')
    plt.xlabel('Category')
    plt.ylim(0, 100) 

    for i, percentage in enumerate(percentages):
        plt.text(i, percentage, f'{percentage:.1f}%', ha='center', va='bottom')

    plt.show()

    print(f"Detected: {with_bbox}/{total} ({with_bbox / total * 100:.1f}%)")