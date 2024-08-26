

# System prompt
You are a helpful assistant and in an annotating videogame images.

# User prompt

## correct_class_label
Evaluate if the object in the bounding box is a {class_label}. Choose one:
    1. Class label is correct: Return the same class label.
    2. Class label is slightly off: Provide the correct class label.
    3. Class label is completely wrong: Provide the correct class label.

    Return JSON:
    {{
        "class_label": "correct or updated label",
        "judgement": "correct|slightly_off|wrong",
        "reasoning": "Brief 10-word explanation"
    }}