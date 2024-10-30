from abc import ABC, abstractmethod
from PIL import Image
import logging

class BaseEmulator(ABC):
    """
    Abstract base class for emulator interfaces.
    Defines the core functionality that all emulator interfaces must implement.
    """
    
    def __init__(self):
        # Configure logger
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(levelname)s - %(name)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    @abstractmethod
    def get_screenshot(self) -> Image.Image:
        """
        Get a screenshot from the emulator.
        
        Returns:
            PIL.Image: Screenshot from the emulator in RGB format
        """
        pass

    @abstractmethod
    def connect_emulator(self):
        """
        Connect to the emulator.
        """
        pass

    @abstractmethod
    def disconnect_emulator(self):
        """
        Disconnect from the emulator.
        """
        pass

    def click(self, x, y, duration=0.1):
        """
        Click on the emulator at the given coordinates.
        """
        pass
    

