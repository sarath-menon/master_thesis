from clicking.segmentation import fetch_data
from clicking.localization.core import LocalizationRequest, LocalizationResp, LocalizationModel
from fastapi import FastAPI
from PIL import Image
import io
import base64
import time
from pydantic import BaseModel

app = FastAPI()
localization_model = LocalizationModel()

@app.get("/")
def root() -> dict:
    print("The FastAPI root endpoint was called.")
    return {"message": fetch_data()}

@app.get("/localization", response_model=LocalizationResp)
async def localization(req: LocalizationRequest):
    return await localization_model.get_localization(req)   

@app.get("/localization/model")
async def get_localization_model():
    model = localization_model.get_model()
    return {"model_name": model.model_name, "model_variant": model.model_id}

@app.post("/localization/model")
async def set_localization_model(req: dict):
    model_name = req.get('model_name')
    model_variant = req.get('model_variant')

    try:
        localization_model.set_model(model_name, model_variant)
        return {"status": "OK"}, 200
    except ValueError as e:
        return {"status": "error", "message": str(e)}, 400


