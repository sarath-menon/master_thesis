from .core import BaseEmulator

class CustomEmulator(BaseEmulator):
    def get_screenshot(self):
        # Your implementation here
        pass

    def connect_emulator(self):
        print("Select an emulator in the dropdown")

    def disconnect_emulator(self):
        print("Select an emulator in the dropdown")

    def click(self, x, y, duration=0.1):
        # Your implementation here
        pass