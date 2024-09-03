

# System prompt
You are a helpful assistant and in an annotating videogame images.

# User prompt

## bbox_overlay
Examine the object enclosed by the red bounding box and assess whether it accurately represents a {object_name}. Provide your response in the following format:

Return a JSON object structured as follows:
{{
    "judgement": "correct" if the label accurately describes the object and it is entirely contained within the bounding box, "bbox_off" if the object is depicted in the image but not completely enclosed by the bounding box, "wrong label" if the label does not correctly describe the object or the object is absent from the image,
    "reasoning": "Provide a precise explanation using exactly 20 words."
}}

Note: Ensure your evaluation is highly precise and specific. Avoid using generalizations or unclear terms.

## bbox_crop

Carefully examine the object in the image and determine if it accurately matches the label {object_name}. Provide your response in the following JSON format:

Return a JSON object structured as follows:
{{
    "judgement": "true" if the label accurately describes the object, "false" if the label does not correctly describe the object or if the object is not present in the image,
    "reasoning": "Provide a clear and concise explanation in 10 words or less."
}}
