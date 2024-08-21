

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
Given a videogame screenshot and a brief description of an object, enhance the description by detailing the object's location, shape, color, and appearance. The description is "{input_description}". 

Return JSON: {{
"expanded_description": "detailed description of the game object in {word_limit} words or less"}}