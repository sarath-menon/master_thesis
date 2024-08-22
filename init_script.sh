#!/bin/bash

# install dependencies
apt update 
apt install libgl1 mesa-utils python3.12-venvgit-lfs just

# install poetry
curl -sSL https://install.python-poetry.org | python3 -
echo 'export PATH="/root/.local/bin:$PATH"' >> ~/.bashrc