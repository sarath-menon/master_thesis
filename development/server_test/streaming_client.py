#%%
import httpx
import asyncio
import nest_asyncio

nest_asyncio.apply()
#%%

async def send_batch_request():
    async with httpx.AsyncClient() as client:
        data = {"items": ["item1", "item2", "item3"]}
        response = await client.post("http://localhost:8000/batch", json=data)
        print(response.json())

async def send_stream_batch_request():
    async with httpx.AsyncClient() as client:
        data = {"items": ["item1", "item2", "item3"]}
        async with client.stream("POST", "http://localhost:8000/stream_batch", json=data) as response:
            async for line in response.aiter_lines():
                print(f"Received: {line}")

#%%
if __name__ == "__main__":
    #asyncio.run(send_batch_request())
    asyncio.run(send_stream_batch_request())