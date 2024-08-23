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

# API client generation

## Using openapi-generator-cli

Generate client

```
mkdir generated
mkdir generated/clicking_client
openapi-generator-cli generate \
     -i http://localhost:8082/openapi.json \
     -g python-fastapi \
     -o ./generated/clicking_client \
     -c ./configs/openapi_generator_config.json
```

Install client 
```
pip install -e ./generated/clicking_client
```

## Using openapi-python-client
```
openapi-python-client generate --url http://localhost:8082/openapi.json  --config ./configs/openapi_generator_config.json  --output-path  ./generated/clicking_client --overwrite
```

Install client 
```
pip install -e ./generated/clicking_client
```

## Using kiota
Doesn't work with OpenAPI specification version '3.1.0' 

```
kiota generate --language python --openapi http://localhost:8082/openapi.json -o ./generated_client --namespace-name clicking_client_kiota
```

# Model packages
## EVF SAM

Generate wheel

```  
python setup.py sdist --dist-dir ../wheels/evf_sam2
```  