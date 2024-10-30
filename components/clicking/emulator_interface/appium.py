from .core import BaseEmulator
from appium import webdriver
from appium.options.common.base import AppiumOptions
from appium.webdriver.common.appiumby import AppiumBy
from appium.webdriver.extensions.action_helpers import ActionHelpers

from selenium.webdriver.common.action_chains import ActionBuilder
from selenium.webdriver.common.actions import interaction
from selenium.webdriver.common.actions.action_builder import ActionBuilder
from selenium.webdriver.common.actions.pointer_input import PointerInput
from selenium.webdriver.common.action_chains import ActionChains
import base64
from datetime import datetime
import time
from PIL import Image
import io
import subprocess
import threading
import os
import signal
import atexit

class AppiumInterface(BaseEmulator):
    _instance = None
    _connection_count = 0
    app_ids = {"settings": "com.apple.Preferences", "monopoly": "com.scopely.monopolygo"}

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            # 1. Create the instance
            cls._instance = super().__new__(cls)
            
            # 2. Initialize BaseEmulator first to set up logger
            BaseEmulator.__init__(cls._instance)
            
            # 3. Now initialize all other attributes
            cls._instance.url = kwargs.get('url', "http://127.0.0.1")
            cls._instance.port = kwargs.get('port', 8210)
            cls._instance.full_url = f"{cls._instance.url}:{cls._instance.port}"
            cls._instance.appium_process = None
            cls._instance._initialized = True
            
            # 4. Start server and continue with other initialization
            cls._instance.start_appium_server()
            
            # Validate launch app
            launch_app = kwargs.get('launch_app', 'monopoly')
            if launch_app not in cls.app_ids:
                raise ValueError(f"Could not find app id for {launch_app}")
            
            # Initialize Appium options
            screenshot_quality = kwargs.get('screenshot_quality', 1)
            mjpeg_quality = kwargs.get('mjpeg_quality', 25)
            mjpeg_framerate = kwargs.get('mjpeg_framerate', 10)
            
            cls._instance.options = AppiumOptions()
            cls._instance.options.load_capabilities({
                "platformName": "iOS",
                "appium:bundleId": cls.app_ids[launch_app],
                "appium:automationName": "XCUITest",
                "appium:udid": "00008030-001104281E02802E",
                "appium:xcodeSigningId": "iPhone Developer",
                "appium:xcodeOrgId": "95Z4N2T99D",
                "appium:updatedWDABundleId": "com.selva123456.WebDriverAgentRunner",
                "appium:includeSafariInWebviews": True,
                "appium:newCommandTimeout": 3600,
                "appium:connectHardwareKeyboard": True,
                "appium:enablePerformanceLogging": True,
                "appium:shouldTerminateApp": True,
                "appium:mjpegServerScreenshotQuality": mjpeg_quality,
                "appium:screenshotQuality": screenshot_quality,
                "appium:mjpegServerFramerate": mjpeg_framerate,
            })
            cls._instance.driver = None
            cls._instance.screen_size = None
            cls._instance.screen_width = None
            cls._instance.screen_height = None

        return cls._instance

    def __init__(self, **kwargs):
        # No initialization needed here as it's all done in __new__
        pass

    def connect_emulator(self):
        try:
            self.logger.info(f"Connecting to emulator on {self.full_url}")
            if not self.driver:
                self.driver = webdriver.Remote(self.full_url, options=self.options)
                self.screen_size = self.driver.get_window_size()
                self.screen_width = self.screen_size['width']
                self.screen_height = self.screen_size['height']
                self.logger.info(f"Device Width and Height: {self.screen_size}")
            self._connection_count += 1
            self.logger.info(f"Connected to emulator. Active connections: {self._connection_count}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to emulator: {e}")
            return False

    def disconnect_emulator(self):
        try:
            self._connection_count -= 1
            if self._connection_count == 0 and self.driver:
                self.driver.quit()
                self.driver = None
                self.logger.info("Disconnected from emulator")
            return True
        except Exception as e:
            self.logger.error(f"Failed to disconnect from emulator: {e}")
            return False

    def swipe_left(self):
        x_left = self.screen_width /9
        x_right = self.screen_width * 8/9
        y_pos = self.screen_height /2

        actions = ActionChains(self.driver)
        actions.w3c_actions.pointer_action.move_to_location(x_left, y_pos)
        actions.w3c_actions.pointer_action.click_and_hold()
        actions.w3c_actions.pointer_action.move_to_location(x_right, y_pos)
        actions.perform()

    def swipe_right(self, x_left, x_right, y_pos):
        actions = ActionChains(self.driver)
        actions.w3c_actions.pointer_action.move_to_location(x_right, y_pos)
        actions.w3c_actions.pointer_action.click_and_hold()
        actions.w3c_actions.pointer_action.move_to_location(x_left, y_pos)
        actions.perform()

    def get_screenshot(self):
        if not self.driver:
            return super().get_screenshot()
        
        image_bytes = self.driver.get_screenshot_as_png()
        return Image.open(io.BytesIO(image_bytes))

    def start_recording(self):
        try:
            self.driver.start_recording_screen()
            self.logger.info("Started screen recording")
        except Exception as e:
            self.logger.error(f"Failed to start recording: {e}")

    def stop_recording(self, filename=None):
        if filename is None:
            filename = f"screen_recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        
        try:
            video_data = self.driver.stop_recording_screen()
            video_bytes = base64.b64decode(video_data)
            with open(filename, "wb") as f:
                f.write(video_bytes)
            self.logger.info(f"Saved recording to {filename}")
        except Exception as e:
            self.logger.error(f"Failed to save recording: {e}")

    def click(self, x, y, duration=0.1):
        screen_width = self.screen_width
        screen_height = self.screen_height

        x = screen_width * x / 100
        y = screen_height * y / 100

        actions = ActionChains(self.driver)
        actions.w3c_actions = ActionBuilder(
            self.driver,
            mouse=PointerInput(interaction.POINTER_TOUCH, "touch")
        )
        actions.w3c_actions.pointer_action.move_to_location(x, y)
        actions.w3c_actions.pointer_action.pointer_down()
        actions.w3c_actions.pointer_action.pause(duration)
        actions.w3c_actions.pointer_action.release()
        actions.perform()

    def start_appium_server(self):
        def run_server():
            self.appium_process = subprocess.Popen(
                ['appium', '-p', str(self.port)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid
            )

        self.server_thread = threading.Thread(target=run_server)
        self.server_thread.daemon = True
        self.server_thread.start()

        time.sleep(5)
        self.logger.info(f"Started Appium server on port {self.port}")
        atexit.register(self.stop_appium_server)

    def stop_appium_server(self):
        if self.appium_process:
            try:
                os.killpg(os.getpgid(self.appium_process.pid), signal.SIGTERM)
            except (ProcessLookupError, OSError) as e:
                self.logger.warning(f"Process already terminated: {e}")
            finally:
                self.appium_process = None
                self.logger.info("Stopped Appium server")

