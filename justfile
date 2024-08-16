
start_server:
    uvicorn bases.clicking.api.core:app --reload --port 8082

install_sam2:
    git clone https://github.com/facebookresearch/segment-anything-2.git
    pip install -e ./segment-anything-2
    rm -rf segment-anything-2

download_sam2_checkpoint VARIANT:
    mkdir -p ./checkpoints/sam2
    wget -q https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_{{VARIANT}}.pt -P ./checkpoints/sam2