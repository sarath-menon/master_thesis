from abc import ABC, abstractmethod
from PIL import Image

class BaseEmulator(ABC):
    """
    Abstract base class for emulator interfaces.
    Defines the core functionality that all emulator interfaces must implement.
    """
    
    @abstractmethod
    def get_screenshot(self) -> Image.Image:
        """
        Get a screenshot from the emulator.
        
        Returns:
            PIL.Image: Screenshot from the emulator in RGB format
        """
        pass
    
    @abstractmethod
    def keypress(self, key: str, duration: int) -> None:
        """
        Send a keypress event to the emulator.
        
        Args:
            key: The key to press
            duration: How long to press the key in milliseconds
        """
        pass

    @abstractmethod
    def pause_emulator(self):
        """
        Pause the emulator.
        """
        pass    

    @abstractmethod
    def resume_emulator(self):
        """
        Resume the emulator.
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
    

