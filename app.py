import streamlit as st
import pandas as pd
import numpy as np
from game_control import GameController
from models import GPT4OModel
import asyncio
import base64

class StreamlitGui:
    def __init__(self):
        self.gc = GameController()

        # Streamlit GUI elements with hot reloading
        st.button('Get screenshot', on_click=self.get_screenshot_callback)
        st.button('Call model', on_click=self.call_model_callback_sync)

        # self.console_output = st.text_area('Console Output', '', key='console_output', height=150)

        self.game_screenshot = None
        self.model = GPT4OModel()

    def get_screenshot_callback(self):
        self.game_screenshot = self.gc.get_screenshot()
        st.image(self.game_screenshot, caption='Game Screenshot')

    def next_image_callback(self):
        pass
    
    def call_model_callback_sync(self):
        asyncio.run(self.call_model_callback())

    async def call_model_callback(self):
        self.game_screenshot = self.gc.get_screenshot()

        if self.game_screenshot is None:
            st.error('No game screenshot to generate response from.')
            return

        base64_image = base64.b64encode(self.game_screenshot).decode('utf-8')

        stream = await self.model.generate_response(base64_image)
        print("Response:", stream)

        response = ""
        async for chunk in stream:
            chunk_content = chunk.choices[0].delta.content
            if chunk_content is not None:
                response = response + chunk_content
                print(chunk_content or "", end="")

        # Update the console output with the response
        st.text_area('Console Output', response, key='console_output', height=150)
