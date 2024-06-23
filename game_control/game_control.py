import pyautogui
import time
from .window_capture import WindowCapture
import requests

class GameController:
    def __init__(self):
        self.game_url = "http://localhost:8086"

        self.score = 0
        self.level = 1
        self.is_game_over = False
        self.is_running = True
        self.screen_width, self.screen_height = pyautogui.size()
        self.window_capture = WindowCapture()
    
    def move_player(self, key, duration=300):
        try:
            response = requests.post(self.game_url + '/move_player', json={'key': key, 'duration': duration})
            response.raise_for_status()  
            return response.status_code
        except requests.RequestException as e:
            print(f"Error fetching game data: {e}")
            return None

    def get_screenshot(self):
        try:
            response = requests.get(self.game_url + '/screenshot')
            response.raise_for_status()  
            return response.content
        except requests.RequestException as e:
            print(f"Error fetching game data: {e}")
            return None

    def click_center(self):
        pyautogui.click(self.screen_width/2, self.screen_height/2)

    def go_to_game_window(self):
        self.click_center()

    def pause_game(self):
        if self.is_running:
            self.click_center()
            pyautogui.hotkey('fn', 'f5')
            self.is_running = False
            print("Game paused!")

    def resume_game(self):
        if not self.is_running:
            self.click_center()
            pyautogui.hotkey('fn', 'f5')
            self.is_running = True
            print("Game resumed!")

    def move_camera(self, direction):
        if direction == "up":
            pyautogui.press('up')
        elif direction == "down":
            pyautogui.press('down')
        elif direction == "left":
            pyautogui.press('left')
        elif direction == "right":
            pyautogui.press('right')
        else:
            print(f"Invalid direction: {direction}")

    def camera_down(self):
        pyautogui.hotkey('fn', 'f7')

    def get_game_state(self):
        return {
            "score": self.score,
            "level": self.level,
            "is_game_over": self.is_game_over
        }
