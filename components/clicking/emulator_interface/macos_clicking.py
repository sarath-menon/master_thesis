from Quartz import CGWindowListCopyWindowInfo, kCGNullWindowID, kCGWindowListOptionAll
import cv2 as cv
import numpy
import time
from PIL import Image, ImageGrab 
import os
import pyautogui
import matplotlib.pyplot as plt
from pydantic import BaseModel
import subprocess
from pynput import mouse

class WindowInfo(BaseModel):
    name: str
    width: int
    height: int
    position: tuple[float, float]
    last_cursor_x: int
    last_cursor_y: int
    id: int

class MacOSInterface:
    def __init__(self, windowName, scale_factor=0.5):
        self.windowName = windowName
        self.window_id_map = {'Ryujinx': '', 'iPhone Mirroring': 'iPhone Mirroring'}
        self.scale_factor = scale_factor

        self.current_window_info = self._getWindowInfo()
        self.mouse_controller = mouse.Controller()

    def _findWindowId(self):
        window_list = CGWindowListCopyWindowInfo(kCGWindowListOptionAll, kCGNullWindowID)

        # ## Iterate through the list and print window details
        # for window in window_list:
        #     if self.windowName == window['kCGWindowOwnerName']:
        #         window_id = window.get('kCGWindowNumber')
        #         window_name = window.get('kCGWindowOwnerName')
        #         window_size = window.get('kCGWindowBounds', {}).get('Width', 'Unknown'), window.get('kCGWindowBounds', {}).get('Height', 'Unknown')
        #         print(f"Window ID: {window_id}, Window Name: {window_name}, Window Size: {window_size}")

        for window in window_list:
            if self.windowName == window['kCGWindowOwnerName']:
                # 'Ryjunix' has many windows, select the window with a name
                if window['kCGWindowName'].strip() != self.window_id_map[self.windowName]:
                    # print('found window id %s' % window.get('kCGWindowNumber'))
                    return window.get('kCGWindowNumber')

        print('unable to find window id')
        return False

    def _getWindowInfo(self) -> WindowInfo | None:
        window_list = CGWindowListCopyWindowInfo(kCGWindowListOptionAll, kCGNullWindowID)
        for window in window_list:
            if self.windowName == window['kCGWindowOwnerName']:
                if self.window_id_map[self.windowName] == window['kCGWindowName']:
                    bounds = window.get('kCGWindowBounds', {})
                    return WindowInfo(
                        name=window['kCGWindowName'],
                        width=bounds.get('Width', 0),
                        height=bounds.get('Height', 0),
                        position=(bounds.get('X', 0), bounds.get('Y', 0)),
                        last_cursor_x=bounds.get('X', 0),
                        last_cursor_y=bounds.get('Y', 0),
                        id=window.get('kCGWindowNumber', 0)
                    )
        print('Unable to find window')
        return None

    def move_cursor(self, x: float, y: float) -> None:
        if not self.current_window_info:
            print("Window information not available")
            return
        
        # Sanity checks
        if not 0 <= x <= 100 or not 0 <= y <= 100:
            print("Invalid percentage values. x and y must be between 0 and 100.")
            return

        window_width = self.current_window_info.width
        window_height = self.current_window_info.height # with border

        border_height = 0

        
        if self.windowName == 'Ryujinx':
            window_x = self.current_window_info.last_cursor_x
            window_y = self.current_window_info.last_cursor_y

            borderless_window_height = int(window_width / (16/9))  # Calculate height for 16:9 aspect ratio
        
            border_height = (window_height - borderless_window_height) / 2
        elif self.windowName == 'iPhone Mirroring':
            window_x = self.current_window_info.position[0]
            window_y = self.current_window_info.position[1]

            borderless_window_height = window_height
        
        else:
            raise ValueError(f"Unknown window name: {self.windowName}")


        # Convert percentage to pixels
        pixel_x = int(window_x + (x / 100) * window_width)
        pixel_y = int(window_y + (y / 100) * borderless_window_height + border_height)
        
        print("Moving cursor to %s, %s" % (pixel_x, pixel_y))
        self.mouse_controller.position = (pixel_x, pixel_y)
        
        # Update current_window_info with new cursor position
        self.current_window_info = WindowInfo(
            name=self.current_window_info.name,
            position=self.current_window_info.position,
            width=self.current_window_info.width,
            height=self.current_window_info.height,
            last_cursor_x=pixel_x,
            last_cursor_y=pixel_y,
            id=self.current_window_info.id
        )
        print(f"Cursor moved to {pixel_x}, {pixel_y}")

    def click(self, x=None, y=None, duration=0.1):
        if x is not None and y is not None:
            self.move_cursor(x, y)
        
        # Activate the window using AppleScript
        script = f'tell application "System Events" to set frontmost of process "{self.windowName}" to true'
        subprocess.run(['osascript', '-e', script], check=True)
        
        # Add a small delay to ensure the window is activated
        time.sleep(0.1)
        
        self.mouse_controller.press(mouse.Button.left)
        time.sleep(duration)
        self.mouse_controller.release(mouse.Button.left)

    def double_click(self, x=None, y=None):
        if x is not None and y is not None:
            self.move_cursor(x, y)

        time.sleep(0.2)
        self.mouse_controller.click(mouse.Button.left, 2)


    def capture_window(self):
        if not self.current_window_info:
            print("Window information not available")
            return None
        
        # Simplified AppleScript to capture specific window
        script = f'''
        tell application "System Events"
            set frontmost of process "{self.windowName}" to true
        end tell
        delay 0.1
        do shell script "screencapture -c -l" & "{self.current_window_info.id}"
        '''
        
        subprocess.run(['osascript', '-e', script], check=True)
        
        # Read clipboard content as image
        clipboard_image = ImageGrab.grabclipboard()
        return clipboard_image
 
