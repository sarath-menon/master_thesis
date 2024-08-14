import re
import json
import base64

def markdown_to_dict(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    # to match markdown headings and their corresponding text
    pattern = r'^(#{1,6})\s*(.*?)\s*\n(.*?)(?=\n#{1,6}\s|\Z)'
    matches = re.findall(pattern, content, re.S | re.M)  

    result = {}
    for match in matches:
        level, heading, text = match
        result[heading.strip()] = text.strip()

    return result

# Open the image file and encode it as a base64 string
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")