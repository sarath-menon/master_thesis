# %%

from google.cloud import storage
import os
# %%
def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    # Optional: set a generation-match precondition to avoid potential race conditions
    # and data corruptions. The request to upload is aborted if the object's
    generation_match_precondition = 0

    blob.upload_from_filename(source_file_name, if_generation_match=generation_match_precondition)

    print(
        f"File {source_file_name} uploaded to {destination_blob_name}."
    )

def upload_many_blobs_with_transfer_manager(
    bucket_name, filenames, source_directory=""
):
    """Upload every file in a list to a bucket, concurrently in a process pool.

    Each blob name is derived from the filename, not including the
    `source_directory` parameter. For complete control of the blob name for each
    file (and other aspects of individual blob metadata), use
    transfer_manager.upload_many() instead.
    """

    from google.cloud.storage import Client, transfer_manager

    storage_client = Client()
    bucket = storage_client.bucket(bucket_name)

    results = transfer_manager.upload_many_from_filenames(
        bucket, filenames, source_directory=source_directory
    )

    for name, result in zip(filenames, results):
        # The results list is either `None` or an exception for each filename in
        # the input list, in order.

        if isinstance(result, Exception):
            print("Failed to upload {} due to exception: {}".format(name, result))
        else:
            print("Uploaded {} to {}.".format(name, bucket.name))

def upload_many_blobs_sequentially(bucket_name, filenames, source_directory=""):
    """Upload every file in a list to a bucket sequentially."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    for filename in filenames:
        try:
            source_file_path = os.path.join(source_directory, filename) if source_directory else filename
            destination_blob_name = filename
            blob = bucket.blob(destination_blob_name)
            blob.upload_from_filename(source_file_path)
            print(f"Uploaded {filename} to {bucket_name}.")
        except Exception as e:
            print(f"Failed to upload {filename} due to exception: {e}")

# %%

bucket_name = "clicking_dataset"
source_file_name = "datasets/resized_media/gameplay_images/fortnite/0.jpg"
destination_blob_name = "ui_dataset"

upload_blob(bucket_name, source_file_name, destination_blob_name)

# %%
bucket_name = "clicking_dataset"
source_directory = "./datasets/resized_media/gameplay_images/fortnite"
destination_prefix = "gameplay_dataset"


filenames = [os.path.join(source_directory, f) for f in os.listdir(source_directory) if os.path.isfile(os.path.join(source_directory, f))]

upload_many_blobs_with_transfer_manager(bucket_name, filenames, source_directory="")

# %%
bucket_name = "clicking_dataset"
source_directory = "datasets/resized_media/gameplay_images/fortnite"
filenames = [f for f in os.listdir(source_directory) if os.path.isfile(os.path.join(source_directory, f))]
upload_many_blobs_sequentially(bucket_name, filenames, source_directory)
#%% Get image using service account

import os
from dotenv import load_dotenv

load_dotenv()

BUCKET_NAME = 'clicking_dataset'  # specify your bucket name here
PREFIX = 'bucket/subfolder'  # specify your prefix here
# assuming you service account key is stored in GOOGLE_APPLICATION_CREDENTIALS
GOOGLE_APPLICATION_CREDENTIALS_FILE = os.getenv('GOOGLE_APPLICATION_CREDENTIALS') 
with open(GOOGLE_APPLICATION_CREDENTIALS_FILE) as f:
    credentials = f.read()
#%% list blobs  
from google.cloud import storage

def list_all_blobs(bucket_name):
    """Lists all the blobs in the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    blobs = bucket.list_blobs()

    for blob in blobs:
        print(blob.name)

# Example usage
bucket_name = "clicking_dataset"
list_all_blobs(bucket_name)

#%%
import os
from google.cloud import storage

def download_blobs_with_prefix_to_disk(bucket_name, prefix, local_dir):
    """Download all blobs with a specific prefix to a local directory."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)

    if not os.path.exists(local_dir):
        os.makedirs(local_dir)

    for blob in blobs:
        local_path = os.path.join(local_dir, blob.name)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        blob.download_to_filename(local_path)
        print(f"Downloaded {blob.name} to {local_path}.")

# Example usage
bucket_name = "clicking_dataset"
prefix = "gameplay_images"
local_dir = "./local_gameplay_images"
download_blobs_with_prefix_to_disk(bucket_name, prefix, local_dir)
#%%
