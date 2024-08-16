from fastapi import FastAPI
from clicking.api.routes.localization import localization_router
from clicking.api.routes.segmentation import segmentation_router

app = FastAPI()

# Root endpoint
@app.get("/")
def root() -> dict:
    return {"message": "Clicking server is running."}

# Include the routes
app.include_router(localization_router, prefix="/localization")
app.include_router(segmentation_router, prefix="/segmentation")