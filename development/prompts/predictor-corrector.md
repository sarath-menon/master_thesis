# Fine grained predicition

Is the object labelled as 'stick' in image actually a stick?. Please recommend a better class label if you can think of one. Otherwise, return the same class label. In both cases, please provide a reasoning in 20 words. Give the output in json format with the following keys: {'object_name': '<object_name>', 'reasoning': '<reasoning>'}

# Detecting wrong prediction

Is the object labelled as 'watertank' in image actually a watertank?. Please recommend a better class label if you can think of one. Otherwise, return the same class label. In both cases, please provide a reasoning in 20 words. Give the output in json format with the following keys: {'object_name': '<object_name>', 'reasoning': '<reasoning>'

# Fixing click point outside object

1. Do you see the letter M/m in the image ?. Respond with:
    - 'is_present': yes/no depending on whether the label is directly on the {object_label}.
    - 'reason': explanation for the positional choice.
    Output should be in JSON format.
2. Assess the position of the label 'y' relative to the {object_label} in the image. Respond with:
    - 'is_overlayed': yes/no depending on whether the label is directly on the {object_label}.
    - 'left/right': position of the label if not overlayed.
    - 'up/down': vertical position of the label if not overlayed.
    - 'reason': explanation for the positional choice.
    Output should be in JSON format.

