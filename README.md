# Usage

## Run model server

To run fastapi server with hot reloading

```
cd bases/clicking/api
uvicorn core:app --reload --port 8082
```

To run robyn server with hot reloading

```
python -m robyn model_server/server.py --dev
```

## Port forwarding

1. start local server
2. start ngrok port forwarding

```
ngrok http http://127.0.0.1:8000

# GCP

## Upload directory to GCP

Using command line

```
gcloud storage cp -r /Users/sarathmenon/Documents/master_thesis/datasets/resized_media/gameplay_images gs://clicking_dataset/
```
-n to not overwrite existing files

```
# Virtual env

## Activation

In cloud instance

```
source /root/master_thesis-1/thesis_env/bin/activate
pip install jupyter
python -m ipykernel install --user --name=thesis_env
```

# Poetry
Testing 

```
poetry run pytest
```