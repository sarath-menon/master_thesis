# System

The image is a gamescene. The game is not mineceaft, although it looks very similar. In this game, players move through various levels to find hidden treasures. Each level is a small maze with obstacles, enemies, and puzzles. Since jumping is not an option, players must think creatively and adjust the camera angles to navigate and solve challenges. The goal is to move safely through each level, avoid hazards, and collect all treasures to advance. The game tests players' problem-solving abilities and awareness of space, offering a fun experience for all ages. Camera movements are key, providing different views of the level to reveal hidden paths, solve puzzles, and locate treasures, thus deepening the strategic and exploratory elements of the game.

The player is indicated by a red box and label. At each timestep, the player has the follwing choices of action:

1. Move in the direction: {forward, backward, left, right}. The direction is relative to the player's current position.
2. Look around using the camera to see the environment and make decisions. The camera can orbit around the player in directions {left, right, up, down}
3. Collect treasures lying beside the player in the current direction.

# User

Decide what to do based on the image and the instructions. Also, give a reasoning for your choice in 30 words. Give the output as JSON code with the following keys:

1. action: {move, look, collect}
2. direction: {forward, backward, left, right}
3. reason: {30 words}

Do not include any preamble or other instructions.

# Movements

1. The image is a gamescene. The game is not minecraft, although it looks very similar. The objective of the game is to explore the scene and collect points. In what direction should the player move? Choose one among the following: {forward, backward, left, right}.
