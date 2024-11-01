#%% 
import io
from PIL import Image
from dataflux_pytorch import dataflux_mapstyle_dataset
import numpy as np
import os


PROJECT_NAME = 'clicking'
BUCKET_NAME = 'clicking_dataset'
PREFIX = 'gameplay_images'
#%% 
# def transform(img_in_bytes): 
#     return np.asarray(
# Image.open(io.BytesIO(img_in_bytes)))

def main():
    dataset = dataflux_mapstyle_dataset.DataFluxMapStyleDataset(
    project_name=PROJECT_NAME,
    bucket_name=BUCKET_NAME,
    config=dataflux_mapstyle_dataset.Config(prefix=PREFIX),
    # data_format_fn=transform,
    )

    sample_object = dataset.objects[0]
    # Learn about the name and the size (in bytes) of the object.
    name = sample_object[0]
    size = sample_object[1]

    print(name, size)

if __name__ == '__main__':
    main()

# %%
dataset = dataflux_mapstyle_dataset.DataFluxMapStyleDataset(
    project_name=PROJECT_NAME,
    bucket_name=BUCKET_NAME,
    config=dataflux_mapstyle_dataset.Config(prefix=PREFIX),
    # data_format_fn=transform,
    )

sample_object = dataset.objects[0]
# Learn about the name and the size (in bytes) of the object.
name = sample_object[0]
size = sample_object[1]

print(name, size)
#%%
dataset