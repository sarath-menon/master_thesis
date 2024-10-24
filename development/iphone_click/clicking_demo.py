#%%
from Quartz import (
    CGWindowListCopyWindowInfo, kCGNullWindowID, kCGWindowListOptionAll,
    CGWindowListCreateImage, CGRectNull, kCGWindowListOptionIncludingWindow,
    kCGWindowImageDefault, CGImageGetWidth, CGImageGetHeight, CGImageGetBytesPerRow,
    CGDataProviderCopyData, CGImageGetDataProvider
)
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
import Quartz
from PIL import Image
import io

class WindowInfo(BaseModel):
    name: str
    width: int
    height: int
    position: tuple[float, float]
    last_cursor_x: int
    last_cursor_y: int
    id: int

class WindowCapture:
    def __init__(self, windowName='Ryujinx', scale_factor=0.5):
        self.windowName = windowName
        self.scale_factor = scale_factor
        self.current_window_info = self._getWindowInfo()
        self.mouse_controller = mouse.Controller()

    def _findWindowId(self, window_name: str):
        window_list = CGWindowListCopyWindowInfo(kCGWindowListOptionAll, kCGNullWindowID)

        # ## Iterate through the list and print window details
        # for window in window_list:
        #     if self.windowName == window['kCGWindowOwnerName']:
        #         window_id = window.get('kCGWindowNumber')
        #         window_name = window.get('kCGWindowOwnerName')
        #         window_size = window.get('kCGWindowBounds', {}).get('Width', 'Unknown'), window.get('kCGWindowBounds', {}).get('Height', 'Unknown')
        #         print(f"Window ID: {window_id}, Window Name: {window_name}, Window Size: {window_size}")

        for window in window_list:
            if window_name == window['kCGWindowOwnerName']:
                # 'Ryjunix' has many windows, select the window with a name
                if window['kCGWindowName'].strip() != "":
                    # print('found window id %s' % window.get('kCGWindowNumber'))
                    return window.get('kCGWindowNumber')

        print('unable to find window id')
        return False

    def _getWindowInfo(self) -> WindowInfo | None:
        window_list = CGWindowListCopyWindowInfo(kCGWindowListOptionAll, kCGNullWindowID)
        for window in window_list:
            if self.windowName == window['kCGWindowOwnerName']:
                if self.windowName == window['kCGWindowName']:
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
        
        window_x = self.current_window_info.last_cursor_x
        window_y = self.current_window_info.last_cursor_y
        window_width = self.current_window_info.width
        window_height = self.current_window_info.height

        print(f"window_width: {window_width}, window_height: {window_height}")
        
        # window_height = int(window_width / (16/9))  # Calculate height for 16:9 aspect ratio
        
        # border_height = (actual_window_height - window_height) / 2

        print("Screen aspect ratio: %s" % (window_height/ window_width ))
        # print("Border height: %s" % border_height)
        
        # Convert percentage to pixels
        pixel_x = int(window_x + (x / 100) * window_width)
        pixel_y = int(window_y + (y / 100) * window_height)

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

# Usage example
#%%
window_capture = WindowCapture(windowName='iPhone Mirroring')

# hamburger menu
x=100
y=100

# go button
x=50.5
y=81.5

# # ring
# x=7.2
# y=75.2

# # hamburger menu
# x=91
# y=7.3

x_offset = 0
y_offset = 5

# offset to account for the hidden border
if x > 50:
    x-=x_offset
else:
    x+=x_offset
if y < 50:
    y+=y_offset
window_capture.click(x, y)

# %% 
window_capture._getWindowInfo()
# %%
screenshot = window_capture.capture_window()
screenshot
# %%
import numpy as np
def crop_image(image_path, start_x=0, end_x=None, start_y=0, crop_height=None, target_width=None):
    """
    Process screenshot with custom cropping and optional scaling
    
    Args:
        image_path: Path to the image file
        start_x: Left crop position
        end_x: Right crop position (if None, uses full width minus start_x)
        start_y: Starting y coordinate for crop
        crop_height: Height of the crop area
        target_width: Desired final width in pixels (maintains aspect ratio if specified)
    """
    img = Image.open(image_path)
    img = img.convert('RGB')
    
    # Handle right side cropping
    if end_x is None:
        end_x = img.width - start_x
    
    # Use full height if not specified
    if crop_height is None:
        crop_height = img.height - start_y
    
    # Perform the crop
    img = img.crop((start_x, start_y, end_x, start_y + crop_height))
    
    # Scale if target width is specified
    if target_width:
        aspect_ratio = img.width / img.height
        target_height = int(target_width / aspect_ratio)
        img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)

    print(f"Final image resolution: {img.size}")
    print(f"Final aspect ratio: {img.height/img.width:.3f}")
    return img

# Example usage:
# img = crop_image('./development/iphone_click/screenshot.png', start_x=120, start_y=145, target_width=1000, crop_height=1630)
img = crop_image('./development/iphone_click/screenshot.png', start_x=0, start_y=5, target_width=1000, crop_height=1625)
plt.imshow(img)
# img.save('./development/iphone_click/screenshot_cropped.png', 'PNG')
# %%
