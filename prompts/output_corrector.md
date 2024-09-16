

# System prompt
You are a helpful assistant and an expert annotator of videogame images.

# User prompt

## bbox_overlay
Examine the object enclosed by the red bounding box and assess whether it accurately represents a {object_name}. Provide your response in the following format:

Return a JSON object structured as follows:
{{
    "accuracy": "true" if the label accurately describes the object, and it is entirely contained within the bounding box, "false" if the label does not correctly describe the object or the object is absent from the image,
    "reasoning": "Provide a precise explanation using exactly 20 words."
}}

## crop

Examine the object in the videogame screenshot and answer the following questions:
1. Does the object match the label {object_name}?
2. Is the object {object_name} fully visible in the image?

Ignore the realism of the image. Focus only on the visibility and accuracy of the object. Provide your response in the following JSON format:

Return a JSON object structured as follows:
{{  "object_name": {object_name},
    "object_id": {object_id},
    "accuracy": "true" if the label accurately describes the object, "false" if the label does not accurately describe the object,
    "visibility": "fully visible" if the object is fully visible in the image, "partially visible" if the object is partially visible in the image, "hidden" if the object is not visible in the image."
    "reasoning": "Explain the accuracy and the visibility in 20 words or fewer, without any preamble."
}}