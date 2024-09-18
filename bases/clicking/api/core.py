from fastapi import FastAPI
from clicking.api.routes.vision_model import vision_model_router
from clicking.api.exceptions import add_exception_handlers
from starlette.middleware.gzip import GZipMiddleware

app = FastAPI()
app.add_middleware(GZipMiddleware, minimum_size=1000)
# add_exception_handlers(app)

# Root endpoint
@app.get("/")
def root() -> dict:
    return {"message": "Clicking server is running."}

# Include the routes
app.include_router(vision_model_router, prefix="/vision_model")

