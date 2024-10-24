from PIL import Image, ImageGrab 
import numpy as np
from .core import BaseEmulator
import subprocess
from .macos_clicking import MacOSInterface
from ..common.image_utils import crop_image

class IphoneMirrorInterface(BaseEmulator):
    _instance = None
    _connection_count = 0

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.macos_interface = MacOSInterface(windowName='iPhone Mirroring')

    def keypress(self, key):
        pass

    def get_screenshot(self):
        img = self.macos_interface.capture_window()
        # # scale image down to 1/2 resolution
        # img = img.resize((img.width // 2, img.height // 2))

        # crop image to remove hidden border
        img = crop_image(img, start_x=120, start_y=150, target_width=1000, crop_height=1625)
        print(f"Image resolution: {img.width}x{img.height}")
        return img

    def pause_emulator(self):
        pass

    def resume_emulator(self):
        pass

    def connect_emulator(self):
        pass

    def disconnect_emulator(self):
        pass
    
    def click(self, x, y, duration=0.1):
        # offset to account for the hidden border
        y_offset = 5

        if y < 50:
            y+=y_offset
            
        self.macos_interface.click(x, y, duration)
   

