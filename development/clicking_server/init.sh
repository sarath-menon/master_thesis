#!/bin/bash

# inssall sam2
git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2; pip install -e .
cd ..

# install pip requirements
git clone https://github.com/sarath-menon/clicking_server.git
pip install -r requirements.txt