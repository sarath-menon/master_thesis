#%%
from appium import webdriver
from appium.options.common.base import AppiumOptions
from appium.webdriver.common.appiumby import AppiumBy
from appium.webdriver.extensions.action_helpers import ActionHelpers

from selenium.webdriver.common.action_chains import ActionBuilder
from selenium.webdriver.common.actions import interaction
from selenium.webdriver.common.actions.action_builder import ActionBuilder
from selenium.webdriver.common.actions.pointer_input import PointerInput

#%% For W3C actions
from selenium.webdriver.common.action_chains import ActionChains
options = AppiumOptions()
options.load_capabilities({
	"platformName": "iOS",
	"appium:bundleId": "com.apple.Preferences",
	"appium:automationName": "XCUITest",
	"appium:udid": "00008030-001104281E02802E",
	"appium:xcodeSigningId": "iPhone Developer",
	"appium:xcodeOrgId": "95Z4N2T99D",
	"appium:updatedWDABundleId": "com.selva123456.WebDriverAgentRunner",
	"appium:showXcodeLog": "true",
	"appium:includeSafariInWebviews": True,
	"appium:newCommandTimeout": 3600,
	"appium:connectHardwareKeyboard": True,
	"appium:enablePerformanceLogging": True,
    "appium:shouldTerminateApp": True,
    "appium:recordVideo": True,
    "appium:videoScale": "1.0",
    "appium:videoType": "h264",
    "appium:videoFps": 30,
    "appium:showTaps": True,  # This will show touch indicators
})

def swipe_left(driver, x_left, x_right, y_pos):
  actions = ActionChains(driver)
  actions.w3c_actions.pointer_action.move_to_location(x_left, y_pos)
  actions.w3c_actions.pointer_action.click_and_hold()
  actions.w3c_actions.pointer_action.move_to_location(x_right, y_pos)
  actions.perform()

def swipe_right(driver, x_left, x_right, y_pos):
  actions = ActionChains(driver)
  actions.w3c_actions.pointer_action.move_to_location(x_right, y_pos)
  actions.w3c_actions.pointer_action.click_and_hold()
  actions.w3c_actions.pointer_action.move_to_location(x_left, y_pos)
  actions.perform()
#%%

driver = webdriver.Remote("http://127.0.0.1:8210", options=options)
# set up swiping
deviceSize = driver.get_window_size()
print("Device Width and Height : ",deviceSize)
screenWidth = deviceSize['width']
screenHeight = deviceSize['height']
#%%

start_x = screenWidth/9
start_y = screenHeight/2

end_x = screenWidth*8/9
end_y = screenHeight/2


actions = ActionChains(driver)
# override as 'touch' pointer action
actions.w3c_actions = ActionBuilder(driver, mouse=PointerInput(interaction.POINTER_TOUCH, "touch"))
actions.w3c_actions.pointer_action.move_to_location(start_x, start_y)
actions.w3c_actions.pointer_action.pointer_down()
actions.w3c_actions.pointer_action.pause(2)
actions.w3c_actions.pointer_action.move_to_location(end_x, end_y)
actions.w3c_actions.pointer_action.release()
actions.perform()
#%%

swipe_left(driver, start_x, end_x, start_y)

# %%
import matplotlib.pyplot as plt

import numpy as np
from PIL import Image
import io

# Convert PNG bytes to PIL Image and then to numpy array
image_bytes = driver.get_screenshot_as_png()
image = Image.open(io.BytesIO(image_bytes))
image_array = np.array(image)

plt.imshow(image_array)
plt.axis('off')
plt.grid(False)
plt.show()
# %%
# screenshot
image.save("screenshot.png")
# %%

# screen record
# ... existing imports ...
from datetime import datetime
import time
# Add these capabilities to your options.load_capabilities():
# options.load_capabilities({
#     # ... your existing capabilities ...
#     "appium:enablePerformanceLogging": True,
#     "appium:shouldTerminateApp": True,
#     "appium:recordVideo": True,
#     "appium:videoScale": "1.0",
#     "appium:videoType": "h264",
#     "appium:videoFps": 30,
#     "appium:showTaps": True,  # This will show touch indicators
# })

def start_recording(driver):
    try:
        driver.start_recording_screen()
        print("Started screen recording")
    except Exception as e:
        print(f"Failed to start recording: {e}")

def stop_recording(driver, filename=None):
    if filename is None:
        filename = f"screen_recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    
    try:
        import base64
        video_data = driver.stop_recording_screen()
        video_bytes = base64.b64decode(video_data)
        with open(filename, "wb") as f:
            f.write(video_bytes)
        print(f"Saved recording to {filename}")
    except Exception as e:
        print(f"Failed to save recording: {e}")

# Usage example (add this where you want to record):
start_recording(driver)

# # Your test actions here
# swipe_left(driver, start_x, end_x, start_y)

time.sleep(3)

# Stop and save the recording
stop_recording(driver, "swipe_test.mp4")
# %%
driver.quit()