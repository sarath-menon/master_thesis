from clicking.segmentation import fetch_data
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root() -> dict:
    print("The FastAPI root endpoint was called.")

    return {"message": fetch_data()}