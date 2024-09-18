from fastapi import FastAPI, BackgroundTasks, Response
from pydantic import BaseModel
from typing import List
import asyncio
import httpx
from fastapi.responses import StreamingResponse

app = FastAPI()

class BatchRequest(BaseModel):
    items: List[str]

class BatchResponse(BaseModel):
    results: List[str]

async def process_item(item: str) -> str:
    await asyncio.sleep(1)  # Simulate processing time
    return f"Processed: {item}"

@app.post("/batch", response_model=BatchResponse)
async def batch_endpoint(request: BatchRequest, background_tasks: BackgroundTasks):
    results = []
    for item in request.items:
        result = await process_item(item)
        results.append(result)
    return BatchResponse(results=results)

async def stream_items(items: List[str]):
    for item in items:
        result = await process_item(item)
        yield f"{result}\n"
x
@app.post("/stream_batch")
async def stream_batch_endpoint(request: BatchRequest):
    return StreamingResponse(stream_items(request.items), media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)