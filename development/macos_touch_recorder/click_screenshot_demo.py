#%%
from Quartz import (
    CGWindowListCopyWindowInfo, kCGNullWindowID, kCGWindowListOptionAll,
    CGWindowListCreateImage, CGRectNull, kCGWindowListOptionIncludingWindow,
    kCGWindowImageDefault, CGImageGetWidth, CGImageGetHeight, CGImageGetBytesPerRow,
    CGDataProviderCopyData, CGImageGetDataProvider
)
from PIL import  ImageGrab 
import matplotlib.pyplot as plt
from pydantic import BaseModel
import subprocess
from pynput import mouse
import Quartz
from PIL import Image
import io
import time

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
        self.screenshot_count = 0
        self.listener = None
        self.save_dir = 'screenshots'

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

    def on_click(self, x, y, button, pressed):
        if button == mouse.Button.left and pressed:
            click_time = int(time.time() * 1000)  # Get timestamp in milliseconds
            print(f'Click detected at: {click_time}ms')
            
            image = self.capture_window()
            if image:
                capture_time = int(time.time() * 1000)  # Get timestamp after capture
                self.screenshot_count += 1
                image.save(f'{self.save_dir}/ screenshot_{self.screenshot_count}.png')

                print(f'Capture delay: {capture_time - click_time}ms')
                print(f'Saved {self.save_dir}/screenshot_{self.screenshot_count}.png')

    def start_listening(self):
        self.listener = mouse.Listener(on_click=self.on_click)
        self.listener.start()
        print("Started listening for clicks. Press Ctrl+C to stop.")
        try:
            self.listener.join()
        except KeyboardInterrupt:
            self.stop_listening()

    def stop_listening(self):
        if self.listener:
            self.listener.stop()
            print("\nStopped listening for clicks")

#%% Usage example
window_capture = WindowCapture(windowName='iPhone Mirroring')
window_capture.start_listening()