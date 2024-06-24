import pyautogui
import time
import requests

class GameController:
    def __init__(self):
        self.game_url = "http://localhost:8086"

        self.score = 0
        self.level = 1
        self.is_game_over = False
        self.is_running = True
        self.screen_width, self.screen_height = pyautogui.size()
    
    def move_player(self, direction, duration=500):
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
        pyautogui.moveTo(self.screen_width/2, self.screen_height/2)
        pyautogui.mouseDown()
        time.sleep(0.1)
        pyautogui.mouseUp()

    def go_to_game_window(self):
        self.click_center()

    def pause_game(self):
        try:
            response = requests.post(self.game_url + '/pause')
            response.raise_for_status()  
            return response.status_code
        except requests.RequestException as e:
            print(f"Error pausing game: {e}")

    def resume_game(self):
        try:
            response = requests.post(self.game_url + '/resume')
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
        try:
            response = requests.post(self.game_url + '/orbit_camera', json={'key': key, 'duration': duration})
            response.raise_for_status()  
            return response.status_code
        except requests.RequestException as e:
            print(f"Error fetching game data: {e}")
            return None

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
        
