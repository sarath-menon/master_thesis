FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

# install dependencies
RUN apt update
RUN apt-get install -y libgl1-mesa-glx  git-lfs 

ARG wheel=clicking_server-0.1.0-py3-none-any.whl

RUN python -m pip install --upgrade pip

WORKDIR /workspace

COPY wheels /workspace/wheels
COPY ./dist/$wheel /workspace/wheels/$wheel

RUN pip install --no-cache-dir --upgrade /workspace/wheels/$wheel

# install wheels
RUN pip install ./wheels/evf_sam2_cuda/evf_sam-1.0-py3-none-any.whl
RUN pip install ./wheels/sam2_cuda/SAM_2-1.0-py3-none-any.whl

# # download model weights
# RUN git clone https://huggingface.co/YxZhang/evf-sam2 checkpoints/evf_sam2

CMD ["uvicorn", "clicking.api.core:app", "--host", "0.0.0.0", "--port", "8082"]
