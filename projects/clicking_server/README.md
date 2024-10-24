# Usage

## Docker

Generate wheel
```
poetry build-project --directory projects/clicking_server
```

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
<!-- docker tag clicking_server sarathmenon1999/clicking_server  -->
docker tag clicking_server:latest sarathmenon1999/clicking_server:v0.1
docker push sarathmenon1999/clicking_server:v0.1
```

## Hugging Face 

High speed model download 

```
huggingface-cli download allenai/Molmo-7B-D-0924
```

## molmo specific 

tensorflow cpu install

```
pip list --format=freeze | grep '^tensorflow' | cut -d= -f1 | xargs -n1 pip uninstall -y
pip install tensorflow-cpu 
```
