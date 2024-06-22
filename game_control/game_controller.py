import pyautogui

class GameController:
    def __init__(self):
        self.score = 0
        self.level = 1
        self.is_game_over = False
        self.is_running = True
        self.screen_width, self.screen_height = pyautogui.size()

    def click_center(self):
        pyautogui.click(self.screen_width/2, self.screen_height/2)

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

    def end_game(self):
        self.is_game_over = True
        print(f"Game over! Final score: {self.score}, Final level: {self.level}")

    def get_game_state(self):
        return {
            "score": self.score,
            "level": self.level,
            "is_game_over": self.is_game_over
        }
