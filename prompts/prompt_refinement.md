# System prompt
You are a helpful assistant and an expert videogame player.

# User prompt

## OBJECTS_LIST_TO_DESCRIPTIONS
Given a videogame screenshot and a list of objects, provide a detailed description for each object. The description should include the object's location, shape, color, and appearance. Output the result as a JSON object with the key "objects". Each entry in the JSON object should contain:
- "description": "A detailed description of the object's location, shape, color, and appearance."
- "reasoning": "A concise 10-word explanation of the object's significance."

## IMAGE_TO_OBJECTS_LIST
Review the videogame screenshot to identify interactive game objects (e.g., doors, chests). Follow these rules strictly:

1. Exclude the following from your identification: playable and non-playable characters, UI elements (buttons, menus, information displays, status bars), and pervasive background elements (grass, fog, sky, trees, rocks, bushes).
2. Evaluate the visibility of each object in the screenshot. Classify them based on the ease of identification:
    - EASY_TO_IDENTIFY: Objects clearly visible and easily distinguishable.
    - MODERATELY_DIFFICULT_TO_IDENTIFY: Objects that are somewhat obscured or blend slightly with the background.
    - HARD_TO_IDENTIFY: Objects that are very small, largely hidden, or blend significantly with the background.
3. Find at least 3 objects that are HARD_TO_IDENTIFY
4. For each object, provide a brief rationale (up to 10 words) for its classification.

Output the information as a JSON list under the key "objects", with each entry including:
- "name": "object name (up to 2 words)"
- "description": "brief details of the object (up to 10 words)"
- "category": "visibility category",
- "reasoning": "rationale for classification (up to 10 words)"

## IMAGE_TO_OBJECT_DESCRIPTIONS
Examine the videogame screenshot to identify crucial game objects for each of the following categories:

1. **Game Asset**: Identify as many game assets as possible. Strictly exclude the following:
   - player characters
   - UI elements (such as buttons, menus, status bars, and information displays)
   - common background elements (such as grass, sky, walls, floors, rocks, and bushes)
2. **Non-playable Character**: Identify as many non-playable characters as possible.
3. **Information Display**: Identify all objects that provide game state information (e.g., health bar, ammo count, minimap, score, notification pop-ups).

<!-- 2. Navigation Controls: Elements that help players navigate the game or menus (e.g., directional pad, joystick, back button, menu button). Identify 3 objects or less.

2. Interactive UI Elements: UI elements that players interact with to perform actions (e.g., action buttons like jump or shoot, inventory slots, dialogue options). Identify 3 objects or less. -->

For each object, provide:
- Name: A name for the object in 2 words or less.
- Description: A detailed description of the object's location, shape, color, and appearance, using no more than {description_length} words. Use full sentences.
- Category: The category the object belongs to (Game Asset, Navigation Control, Information Display, Interactive UI Element).
- Reasoning: A concise 10-word explanation of why the object is important for interaction.

Format the output as a JSON list under the key "objects", with each entry containing:
- "name": "object name in 2 words or less",
- "description": "details of the object including location, shape, color, and appearance",
- "category": "category name",
- "reasoning": "10-word explanation of its significance"