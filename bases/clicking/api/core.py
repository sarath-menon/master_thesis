from fastapi import FastAPI
from clicking.api.routes.localization import localization_router
from clicking.segmentation import fetch_data

app = FastAPI()

# Root endpoint
@app.get("/")
def root() -> dict:
    print("The FastAPI root endpoint was called.")
    return {"message": fetch_data()}

# Include the localization router in the main app
app.include_router(localization_router, prefix="/localization")