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
- "name": "object name (up to {object_name_limit} words)"
- "description": "brief details of the object (up to {description_length} words)"
- "category": "visibility category",
- "reasoning": "rationale for classification (up to 10 words)"

## IMAGE_TO_OBJECT_DESCRIPTIONS
Examine the videogame screenshot to identify crucial game objects for each of the following categories:

1. **Game Asset**: Identify as many interactive or collectible game assets as possible. Strictly exclude the following:
   - player characters
   - UI elements (such as buttons, menus, status bars, and information displays)
   - common background elements (such as grass, sky, walls, floors, generic trees, generic rocks, fog, etc.)
   - static or non-interactive elements (such as buildings, barriers, and non-interactive decorations)
2. **Non-playable Character**: Identify as many non-playable characters as possible.
3. **Information Display**: Identify all objects that provide game state information (e.g., health bar, ammo count, minimap, score, notification pop-ups).

For each object, provide:
- Name: A name for the object in {object_name_limit} words or less.
- Description: Provide a detailed description of the object's location on the screen, its shape, color, and appearance. Use no more than {description_length} words. Format the description as follows: <object_name> located at the <object_location> of the screen with a <shape>, <color> and <appearance>.
- Category: The category the object belongs to one of the above.
- Reasoning: A concise 10-word explanation of why the object is important for interaction.

Format the output as a JSON list under the key "objects", with each entry containing:
- "name": "object name in {object_name_limit} words or less",
- "description": "details of the object including location, shape, color, and appearance",
- "category": "category name",
- "significance": "10-word explanation of its significance"