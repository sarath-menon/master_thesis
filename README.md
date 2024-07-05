# Usage

## Run model server

To run fastapi server with hot reloading

```
cd model_server
uvicorn main:app --reload --port 8082
```

To run robyn server with hot reloading

```
python -m robyn model_server/server.py --dev
```

## Port forwarding

1. start local server

```
uvicorn main:app --reload
```

2. start ngrok port forwarding

```
ngrok http http://127.0.0.1:8000
```
