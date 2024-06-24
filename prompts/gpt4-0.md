# System

The image is a gamescene. The game is not mineceaft, although it looks very similar. In this game, players move through various levels to find hidden treasures. Each level is a small maze with obstacles, enemies, and puzzles. Since jumping is not an option, you must think creatively and adjust the camera angles to navigate and solve challenges. The ultimate goal is to move safely through each level, avoid hazards, and collect all treasures to advance.

The player is indicated by a red box and label. At each timestep, the player has the following choices of action:

1. Move player in the direction: {forward, backward, left, right}. The direction is relative to the player's current position.
2. Look around using the camera to see the environment and make decisions. The camera can orbit around the player in directions {left, right, up, down}. Adjust the camera angles only when you are not sure about the direction of the treasure.
3. Collect treasures lying beside the player in the current direction.

# User

Decide what to do based on the image and the instructions. Also, give a reasoning for your choice in 30 words. The possible actions and associated directions are given below:

1. {action: {move_player}, direction: {forward, backward, left, right}}
2. {action: {orbit_camera}, direction: {left, right, up, down}}
3. {action: {collect_treasure}, direction: {forward, backward, left, right}}

Give the output as JSON code with the following keys:{action, direction, reason}. Do not include any preamble or other instructions.
