
start_server PORT='8082':
    uvicorn core:app --reload --port {{PORT}} --app-dir bases/clicking/api --reload-dir bases/clicking/api

install_sam2:
    git clone https://github.com/facebookresearch/segment-anything-2.git
    pip install -e ./segment-anything-2
    rm -rf segment-anything-2

download_sam2_checkpoint VARIANT:
    mkdir -p ./checkpoints/sam2
    wget -q https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_{{VARIANT}}.pt -P ./checkpoints/sam2

install_openapi_client:
    mkdir -p ./generated/clicking_client
    openapi-python-client generate --url http://localhost:8082/openapi.json  --config ./configs/openapi_generator_config.json  --output-path  ./generated/clicking_client --overwrite
    pip install -e ./generated/clicking_client

poetry_shell:
    bash -c 'source ~/.bashrc && poetry shell'