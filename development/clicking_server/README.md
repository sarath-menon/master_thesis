# Without docker

## git config
```
git config --global user.name "sarath-menon"
git config --global user.email "sarathmenon.downey@gmail.com"
```

## Run model server

Install python dependencies

```
pip install -r requirements.txt
```

To run fastapi server with hot reloading

```
uvicorn main:app --reload --port 8082
```

```
python main.py
```

# With docker

```
docker build . -t deleteme  
```


TO run the server:

```
docker run -p 8082:8082 deleteme
```
## Install dependencies
```
apt-get install -y libgl1-mesa-glx
```                              

Add python kernel to jupyter
```    
python -m ipykernel install --user --name python3 --display-name "Python 3"
jupyter kernelspec list
```    
## Download sam2 weights

```
mkdir checkpoints
cd checkpoints
```

Install SAM2
```
git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2; pip install -e .
```

```
mkdir checkpoints
wget -q https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt -P ./checkpoints
```

```
wget -q https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt -P {HOME}/checkpoints
wget -q https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt -P {HOME}/checkpoints
wget -q https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt -P {HOME}/checkpoints
wget -q https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt -P {HOME}/checkpoints
```