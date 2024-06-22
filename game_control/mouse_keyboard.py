from pymouse import PyMouse
from pykeyboard import PyKeyboard
import signal
import sys
from game_controller import GameController
import time
import pyautogui

game_controller = GameController()

# currentMouseX, currentMouseY = pyautogui.position()
# print(f"Current mouse position: {currentMouseX}, {currentMouseY}")


game_controller.pause_game()
time.sleep(2)
game_controller.resume_game()

