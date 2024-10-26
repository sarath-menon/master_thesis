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
	"appium:connectHardwareKeyboard": True
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
# save image
image.save("screenshot.png")
# %%
