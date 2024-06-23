from game_controller import GameController
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

game_controller = GameController()

# game_controller.resume_game()
game_controller.go_to_game_window()
game_controller.move_player('left', 1)


img = game_controller.get_screenshot()
plt.imshow(img)
plt.axis('off')
plt.show()



# game_controller.pause_game()
