from PIL import Image, ImageGrab 
import numpy as np
from .core import BaseEmulator
import subprocess
from .macos_clicking import MacOSInterface

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
        # scale image down to 1/2 resolution
        img = img.resize((img.width // 2, img.height // 2))
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

        # add offset to account for the hidden border
        y += 6.5
        x -= 7
        self.macos_interface.click(x, y, duration)
   

