# User prompt

## clickpoint_input
The image is a game screenshot. Describe the object at the coordinates:
{{       
    "x": {x_percent:.1f},
    "y": {y_percent:.1f}
}}. Follow these instructions strictly:
1. Make sure that you describe the exact object at this point. 
2. If the object has a name or a label, include it in the description. 
3. If the object is one among a grid of objects, give row column indices of the object, with the top left item as the origin
  


## clickpoint_input_json
The image provided is a game screenshot. Your task is to describe the object located at the specified coordinates:
{{       
    "x": {x_percent:.1f},
    "y": {y_percent:.1f}
}}.

Instructions:
1. Identify and describe the exact object at the given coordinates.
2. Provide the description in JSON format with the following keys:
    - "type": Specify whether the object is a "game asset" or a "UI element".
    - "description": Provide a description of the exact object at the specified coordinates. If the object has a name or a label, include it in the description.