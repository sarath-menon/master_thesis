# System prompt
You are a helpful assistant and an expert videogame player.

# User prompt
## prompt_to_class_label
Analyze the videogame screenshot. For the action "{action}":
    1. Identify the game object to selected to execute the action. 
    2. Provide a brief 10-word reasoning.
    
    Return JSON: {{
        "class_label": "object to click",
        "reasoning": "10-word explanation"
    }}

## prompt_expansion
Given a videogame screenshot and a brief description of an object, enhance the description by detailing the object's location, shape, color, and appearance. The description is "{input_description}". Give the description in {word_limit} words or less.


## image_to_class_label
Examine the videogame screenshot to identify crucial game objects for each of the following categories:

1. Game Assets: Objects within the game world that can be interacted with (e.g., door, chest, button). Identify exactly 5 objects.
2. Navigation Controls: Elements that help players navigate the game or menus (e.g., directional pad, joystick, back button, menu button). Identify 3 objects or less.
3. Information Displays: Elements that provide real-time game state information (e.g., health bar, ammo count, minimap, score, notification pop-ups). Identify 3 objects or less.
4. Interactive UI Elements: UI elements that players interact with to perform actions (e.g., action buttons like jump or shoot, inventory slots, dialogue options). Identify 3 objects or less.

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