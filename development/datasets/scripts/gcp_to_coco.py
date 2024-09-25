#%%
import json
from google.cloud import storage
import os

def list_gcs_files_with_prefix(bucket_name: str, prefix: str):
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)
    return [f"https://storage.googleapis.com/{bucket_name}/{blob.name}" for blob in blobs]

def save_image_urls_to_coco_format(bucket_name: str, prefix: str, output_file: str):
    image_urls = list_gcs_files_with_prefix(bucket_name, prefix)
    
    coco_dataset = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    for idx, url in enumerate(image_urls):
        if url.lower().endswith('.jpg'):
            image_info = {
                "id": idx,
                "file_name": url.split("/")[-1],
                "coco_url": url,
                "height": 0,  # Placeholder, update with actual height if available
                "width": 0,   # Placeholder, update with actual width if available
            }
            coco_dataset["images"].append(image_info)
    # Add categories if needed
    # coco_dataset["categories"].append({
    #     "id": 1,
    #     "name": "category_name",
    #     "supercategory": "supercategory_name"
    # })
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(coco_dataset, f, indent=4)
    print(f"COCO dataset JSON saved to {output_file}")

#%%
if __name__ == "__main__":
    bucket_name = 'clicking_dataset'
    prefix = 'annotation_dataset'
    output_file = 'datasets/annotation_dataset/coco_dataset_gcp/result.json'
    save_image_urls_to_coco_format(bucket_name, prefix, output_file)