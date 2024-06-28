import pyautogui
import time
import requests
import websocket
import threading
import time
import io
from PIL import Image
import cv2
import numpy as np
import json
import queue
import atexit

class GameController:
    def __init__(self):
        self.game_url = "http://localhost:8086"

        self.score = 0
        self.level = 1
        self.is_game_over = False
        self.is_running = True
        self.screen_width, self.screen_height = pyautogui.size()

        self.frame_count = 0
        self.start_time = time.time()
        self.image_queue = queue.LifoQueue(maxsize=100)

        # Example variables - replace these with actual values
        self.image_data = b'...'  # This should be your _imageByte data
        self.width = 1920      # Replace with actual image width
        self.height = 1080        # Replace with actual image height

         # Create a separate WebSocket for receiving images
        ws_image = websocket.WebSocketApp("ws:"+ self.game_url + "/stream_websocket",                     
                                           on_message=self.on_image_message,
                                           on_open=self.on_image_open,
                                           on_error=self.on_error,
                                           on_close=self.on_close)
        self.ws_image_thread = threading.Thread(target=ws_image.run_forever)
        self.ws_image_thread.start()
        # Register a function to stop the thread when the program is about to exit
        atexit.register(self.stop_threads)

    def stop_threads(self):
        # Stop the WebSocket thread
        self.ws_image_thread.join()

    def on_image_message(self, ws, message):
        # Create an image from the byte data
        image = Image.frombytes(mode='RGBA', size=(self.width, self.height), data=message)

        # Convert the image to a numpy array only if necessary
        img_np = np.array(image)
        # img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

        if not self.image_queue.full():
            self.image_queue.put(img_np)
        else:
            self.image_queue.queue.clear()

        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        if elapsed_time > 1 and self.frame_count % 10:  # Update FPS every second
            fps = self.frame_count / elapsed_time
            print(f"Receiver FPS: {fps:.2f}")
            self.frame_count = 0
            self.start_time = time.time()

    def on_image_open(self, ws):
        def run(*args):
            # You can send messages to the server here if needed
            # ws.send("Hello Server")
            while True:
                time.sleep(20)  # Keep the connection open
        thread = threading.Thread(target=run)
        thread.daemon = True  # Set thread as daemon so it closes with the main program
        thread.start()

    def on_error(self, ws, error):
        print(f"Error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        print("### closed ###")

    def keypress(self, key, duration):
        try:
            response = requests.post(self.game_url + '/keypress', json={'key': key, 'duration': duration})
            response.raise_for_status()  
            return response.status_code
        except requests.RequestException as e:
            print(f"Error fetching game data: {e}")
            return None 
    
    def move_player(self, direction, duration=3000):
        key = None
        if direction == "forward":
            key = 'W'
        elif direction == "backward":
            key = 'S'
        elif direction == "left":
            key = 'A'
        elif direction == "right":
            key = 'D'
        else:
            print(f"Invalid direction: {direction}")
            return None

        self.keypress(key, duration)
       

    def get_screenshot(self):
        # try:
        #     response = requests.get(self.game_url + '/screenshot')
        #     response.raise_for_status()  
        #     return response.content
        # except requests.RequestException as e:
        #     print(f"Error fetching game data: {e}")
        #     return None

        try:
            if self.image_queue.qsize() > 0:
                print("Image queue size: ", self.image_queue.qsize())
                image = self.image_queue.get()
                return image
            else:
                return None
        except Exception as e:
            print(f"Error getting screenshot: {e}")
            return None

    def click_center(self):
        pyautogui.moveTo(self.screen_width/2, self.screen_height/2)
        pyautogui.mouseDown()
        time.sleep(0.1)
        pyautogui.mouseUp()

    def go_to_game_window(self):
        self.click_center()

    def pause_game(self):
        try:
            response = requests.post(self.game_url + '/pause_game')
            response.raise_for_status()  
            return response.status_code
        except requests.RequestException as e:
            print(f"Error pausing game: {e}")

    def resume_game(self):
        try:
            response = requests.post(self.game_url + '/resume_game')
            response.raise_for_status()  
            return response.status_code
        except requests.RequestException as e:
            print(f"Error resuming game: {e}")

    def orbit_camera(self, direction, duration=300):
        key = None
        if direction == "up":
            key = 'Up'
        elif direction == "down":
            key = 'Down'
        elif direction == "left":
            key = 'Left'
        elif direction == "right":
            key = 'Right'
        else:
            print(f"Invalid direction: {direction}")
            return None
        self.keypress(key, duration)

    def special_action(self, action):
        key = None
        duration = 500

        if action == "jump":
            key = 'B'
        elif action == "throw_hat":
            key = 'Y'
        else:
            print(f"Invalid special action: {action}")
            return None

        self.keypress(key, duration)
        

    def camera_down(self):
        pyautogui.hotkey('fn', 'f7')

    def get_game_state(self):
        return {
            "score": self.score,
            "level": self.level,
            "is_game_over": self.is_game_over
        }
    
    def collect_treasure(self, direction):
        print(f"Collecting treasure in {direction}")
        
