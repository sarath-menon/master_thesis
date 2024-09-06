from clicking_client.types import File
from io import BytesIO

def image_to_http_file(image):
    # Convert PIL Image to bytes and create a File object
    image_byte_arr = BytesIO()
    image.save(image_byte_arr, format='PNG')
    image_file = File(file_name="image.png", payload=image_byte_arr.getvalue(), mime_type="image/png")
    return image_file