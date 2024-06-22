from Quartz import CGWindowListCopyWindowInfo, kCGNullWindowID, kCGWindowListOptionAll
import cv2 as cv
import numpy
import time
from PIL import Image, ImageGrab 
import os
import pyautogui
import matplotlib.pyplot as plt

windowName = "Ryujinx"
scale_factor = 0.5  # Define the scale factor for downsampling

def findWindowId(windowName):
    window_list = CGWindowListCopyWindowInfo(kCGWindowListOptionAll, kCGNullWindowID)

    # # Iterate through the list and print window details
    # for window in window_list:
    #     if windowName == window['kCGWindowOwnerName']:
    #         window_id = window.get('kCGWindowNumber')
    #         window_name = window.get('kCGWindowOwnerName')
    #         window_size = window.get('kCGWindowBounds', {}).get('Width', 'Unknown'), window.get('kCGWindowBounds', {}).get('Height', 'Unknown')
    #         print(f"Window ID: {window_id}, Window Name: {window_name}, Window Size: {window_size}")

    for window in window_list:
        if windowName == window['kCGWindowOwnerName']:
            # 'Ryjunix' has many windows, select the window with a name
            if window['kCGWindowName'].strip() != "":
                print('found window id %s' % window.get('kCGWindowNumber'))
                return window.get('kCGWindowNumber')

    print('unable to find window id')
    return False


def takeScreenshot(windowId):
    windowId = findWindowId(windowId)

    imageFileName = 'screen.png'
    # -x mutes sound, -c copies to clipboard, -l specifies windowId
     # Use screencapture to copy the screen to the clipboard instead of saving to a file
    os.system('screencapture -c -x -l %s' % windowId)

    # Load the image from the clipboard
    img = ImageGrab.grabclipboard()

    if img is not None:
        # Get the original dimensions
        original_width, original_height = img.size

        # Calculate the new dimensions
        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)
        print(f"Original dimensions: {original_width}x{original_height}, New dimensions: {new_width}x{new_height}")

        # Downsample the image
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Crop 50 pixels from the top and bottom
        crop_top = 60
        crop_bottom = 60
        cropped_img = img.crop((0, crop_top, new_width, new_height - crop_bottom))  
        return cropped_img
    else:
        print("No image on clipboard!")
        return None

img = takeScreenshot(windowName)

if img is not None:
    img = img.convert('RGB')  # Convert the image to RGB mode
    img.save('screenshot.jpg', format='JPEG', quality=100)
    plt.imshow(img)
    plt.title(windowName)
    plt.axis('off')
    plt.show()


