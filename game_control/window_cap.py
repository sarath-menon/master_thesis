from Quartz import CGWindowListCopyWindowInfo, kCGNullWindowID, kCGWindowListOptionAll
import cv2 as cv
import numpy
import time
from PIL import Image, ImageGrab 
import os
import pyautogui
import matplotlib.pyplot as plt

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
        img_array = numpy.array(img)
        return img_array
    else:
        print("No image on clipboard!")
        return None
    
windowName = "Ryujinx"
img = takeScreenshot(windowName)

if img is not None:
    plt.imshow(img)
    plt.show()


