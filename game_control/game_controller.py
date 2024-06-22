import pyautogui
import time

class GameController:
    def __init__(self):
        self.score = 0
        self.level = 1
        self.is_game_over = False
        self.is_running = True
        self.screen_width, self.screen_height = pyautogui.size()

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

    # duration is in seconds
    def move_player(self, direction, duration=0.1):
        key = None

        # set key
        if direction == "forward":
            key = 'w'
        elif direction == "backward":
            key = 's'
        elif direction == "left":
            key = 'a'
        elif direction == "right":
            key = 'd'
        else:
            print(f"Invalid direction: {direction}")

        # press key for specific duration
        pyautogui.keyDown(key)
        time.sleep(duration)  # Duration for which the key will be held down
        pyautogui.keyUp(key)

    def camera_down(self):
        pyautogui.hotkey('fn', 'f7')

    def get_game_state(self):
        return {
            "score": self.score,
            "level": self.level,
            "is_game_over": self.is_game_over
        }
