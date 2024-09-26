# Usage

## Docker

To build and run the docker container

```
docker build -t clicking_server .
docker run -d --name clicking_container -p 8082:8082 clicking_server
```

To stop the container

```
docker stop clicking_container
docker rm clicking_container
```

To push container to docker hub
````
docker tag clicking_server sarathmenon1999/clicking_server 
docker push sarathmenon1999/clicking_server
```

## Hugging Face

High speed model download 

```
pip install huggingface_hub[hf_transfer]
huggingface-cli download allenai/Molmo-7B-D-0924
```
