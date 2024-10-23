FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

# install dependencies
RUN apt update

# Install dependencies
RUN apt-get install -y libgl1-mesa-glx  git-lfs

# Upgrade pip
RUN pip3 install --upgrade pip

ARG wheel=clicking_server-0.1.0-py3-none-any.whl

# # Create a virtual environment
# RUN python -m venv /workspace/venv

# # Activate the virtual environment
# ENV PATH="/workspace/venv/bin:$PATH"

# Upgrade pip in the virtual environment
RUN pip install --upgrade pip

WORKDIR /workspace

# Copy the wheel file
COPY ./dist/$wheel /workspace/wheels/$wheel

# Install the wheel in the virtual environment
RUN pip install /workspace/wheels/$wheel

# # install wheels
# RUN pip install --no-cache-dir --upgrade /workspace/wheels/$wheel

# # install wheels
# RUN pip install ./wheels/evf_sam2_cuda/evf_sam-1.0-py3-none-any.whl
# RUN pip install ./wheels/sam2_cuda/SAM_2-1.0-py3-none-any.whl
# RUN pip install ./wheels/$wheel

# # download model weights
# # RUN git clone https://huggingface.co/YxZhang/evf-sam2 checkpoints/evf_sam2
# CMD huggingface-cli download allenai/Molmo-7B-D-0924

# CMD ["uvicorn", "clicking.api.core:app", "--host", "0.0.0.0", "--port", "8082"]

# # Copy the start script into the image
# COPY start.sh /root/onstart.sh

# # Make the script executable
# RUN chmod +x /root/onstart.sh

# # Use the script as the CMD
# CMD ["/root/onstart.sh"]
