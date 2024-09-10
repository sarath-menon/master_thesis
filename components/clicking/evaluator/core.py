from clicking.image_processor.visualization import overlay_bounding_box
from clicking.common.data_structures import PipelineState
from typing import List, Dict
import os
import json
from pydantic import BaseModel, Field
from evaluate import load
from collections import Counter
from typing import List, Dict, Literal
from clicking.common.data_structures import ClickingImage, ValidityStatus
import wandb
import pandas as pd

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
