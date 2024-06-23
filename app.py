
import streamlit as st
import pandas as pd
import numpy as np
from game_control import GameController
import keyboard 

class StreamlitGui:
    def __init__(self):
        self.gc = GameController()

        # Initialize the Streamlit GUI elements with hot reloading
        st.title('Game Image Viewer')
        st.write('This GUI displays images captured from a game.')
        self.image_caption = st.text_input('Enter image caption:')
        st.button('Get screenshot', on_click=self.get_screenshot_callback)
        st.button('Next Image', on_click=self.next_image_callback)

        st.text_area('Console Output', '', key='console_output', height=150)

    def get_screenshot_callback(self):
        img = self.gc.get_screenshot()
        st.image(img, caption='Game Screenshot')

    def next_image_callback(self):
        pass

    
        # self.gc.move_player(event.name)

