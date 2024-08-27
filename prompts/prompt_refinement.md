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
Examine the videogame screenshot to pinpoint up to five crucial interactive objects, which may include both game objects and UI elements. For each object, provide a description that covers its location, shape, color, and appearance, using no more than {description_length} words. Use full sentences to describe the object rather than short phrases. Additionally, include a concise explanation, limited to 10 words, detailing why each object is important for interaction. The output should be formatted as a JSON list under the key "objects", with each entry containing:
- "description": "details of the object including location, shape, color, and appearance",
- "category": "game object or UI element",
- "reasoning": "10-word explanation of its significance"