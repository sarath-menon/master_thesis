
start_server PORT='8082':
    uvicorn core:app --reload --port {{PORT}} --app-dir bases/clicking/api --reload-dir bases/clicking/api

install_wheels:
    pip install ./wheels/evf_sam2/evf_sam-1.0-py3-none-any.whl
    pip install ./wheels/sam2/SAM_2-1.0-py3-none-any.whl

download_weights:
    git clone https://huggingface.co/YxZhang/evf-sam2 checkpoints/evf_sam2

download_sam2_checkpoint VARIANT:
    mkdir -p ./checkpoints/sam2
    wget -q https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_{{VARIANT}}.pt -P ./checkpoints/sam2

install_openapi_client PORT='8082':
    mkdir -p ./generated/clicking_client
    openapi-python-client generate --url http://localhost:{{PORT}}/openapi.json  --config ./configs/openapi_generator_config.json  --output-path  ./generated/clicking_client --overwrite
    pip install -e ./generated/clicking_client

poetry_shell:
    bash -c 'source ~/.bashrc && poetry shell'

test BRICK NAME *ARGS:
    poetry run pytest -p no:warnings test/{{BRICK}}/clicking/{{NAME}} -x -v --tb=short {{ARGS}}

