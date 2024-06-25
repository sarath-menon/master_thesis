import os
from openai import AsyncOpenAI
import asyncio
from . import utils
import json
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import atexit

class GPT4OModel:    
    def __init__(self, prompt_path):
        self.MODEL = "gpt-4o"
        self.PROMPT_PATH = prompt_path
        self.client = AsyncOpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
        )
        self.prompts_dict = utils.markdown_to_dict(self.PROMPT_PATH)
        self.observer = None

        self._start_watcher()

    def _start_watcher(self):
        if self.observer is None:
            event_handler = self._FileChangeHandler(self)
            self.observer = Observer()
            self.observer.schedule(event_handler, path=os.path.dirname(self.PROMPT_PATH), recursive=False)
            self.observer.start()
            atexit.register(self._stop_watcher)

    def _stop_watcher(self):
        if self.observer is not None:
            self.observer.stop()
            self.observer.join()
            self.observer = None

    class _FileChangeHandler(FileSystemEventHandler):
        def __init__(self, model_instance):
            self.model_instance = model_instance

        def on_modified(self, event):
            print("Prompt file modified")
            if event.src_path == self.model_instance.PROMPT_PATH:
                self.model_instance.prompts_dict = utils.markdown_to_dict(self.model_instance.PROMPT_PATH)
            
            print(self.model_instance.prompts_dict)

    async def generate_response(self, base64_image):
        stream = await self.client.chat.completions.create(
            model=self.MODEL,
            messages=[
                {"role": "system", "content": self.prompts_dict['System']},
                {"role": "user", "content": [
                    {"type": "text", "text": self.prompts_dict['User']},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"}
                    }
                ]}
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        return stream

    async def generate_waste(self, base64_image):
        response = await self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a waste classification model"},
                {"role": "user", "content": "return a short json with 3 fields, one called reason"}
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        return response

    async def generate_waste_async(self, base64_image):
        stream = await self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a waste classification model"},
                {"role": "user", "content": "Say oh yeah"}
            ],
            response_format={"type": "json_object"},
            stream=True,
            temperature=0.0,
        )
        return stream

