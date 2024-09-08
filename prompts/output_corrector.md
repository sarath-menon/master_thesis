

# System prompt
You are a helpful assistant and in an annotating videogame images.

# User prompt

## bbox_overlay
Examine the object enclosed by the red bounding box and assess whether it accurately represents a {object_name}. Provide your response in the following format:

Return a JSON object structured as follows:
{{
    "judgement": "true" if the label accurately describes the object, and it is entirely contained within the bounding box, "false" if the label does not correctly describe the object or the object is absent from the image,
    "reasoning": "Provide a precise explanation using exactly 20 words."
}}

## bbox_crop

Examine the object in the image and determine the following:
1. Does it match the label {object_name}?
2. Is the {object_name} fully visible in the image?

Provide your response in the following JSON format:

Return a JSON object structured as follows:
{{  "object_name": {object_name},
    "object_id": {object_id},
    "judgement": "true" if the label accurately describes the object, "false" if the label does not accurately describe the object,
    "visibility": "fully visible" if the object is fully visible in the image, "partially visible" if the object is partially visible in the image, "hidden" if the object is not visible in the image."
    "reasoning": "Explain the judgement and the visibility in 20 words or fewer, without any preamble."
}}