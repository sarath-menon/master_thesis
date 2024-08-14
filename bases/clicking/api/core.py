from clicking.segmentation import fetch_data
from clicking.localization.core import LocalizationRequest, LocalizationResp, get_localization
from fastapi import FastAPI
from PIL import Image
import io
import base64
import time
from pydantic import BaseModel

app = FastAPI()

@app.get("/")
def root() -> dict:
    print("The FastAPI root endpoint was called.")
    return {"message": fetch_data()}

@app.get("/localization", response_model=LocalizationResp)
async def localization(req: LocalizationRequest):
    return await get_localization(req)   