#!/bin/bash

# Clone the repository
git clone https://github.com/facebookresearch/segment-anything-2.git

# Install the package in editable mode
cd segment-anything-2 &&  pip install -e .

# # Clean up
# cd ..
# rm -rf segment-anything-2

echo "Package installed successfully"
