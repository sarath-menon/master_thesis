
import streamlit as st
import pandas as pd
import numpy as np
from game_control import GameController
import keyboard 

class StreamlitGui:
    def __init__(self):
        self.gc = GameController()

        # Streamlit GUI elements with hot reloading
        st.button('Get screenshot', on_click=self.get_screenshot_callback)
        st.text_area('Console Output', '', key='console_output', height=150)

    def get_screenshot_callback(self):
        img = self.gc.get_screenshot()
        st.image(img, caption='Game Screenshot')

    def next_image_callback(self):
        pass

    
        # self.gc.move_player(event.name)

