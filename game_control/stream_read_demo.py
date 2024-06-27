import requests
import shutil
from PIL import Image
from io import BytesIO

def process_frame(frame_data):
    """
    Process the frame data, for example, displaying or saving the frame.
    """
    image = Image.open(BytesIO(frame_data))
    image.show()  # Display the image using the default image viewer

def stream_frames(url):
    """
    Connect to the server and stream frames continuously.
    """
    with requests.get(url, stream=True) as response:
        if response.status_code != 200:
            print("Failed to connect to the server.")
            return

        # Boundary is defined in the server's response header 'content-type'
        boundary = response.headers['content-type'].split('=')[1]
        buffer = b''
        
        for chunk in response.iter_content(chunk_size=1024):
            buffer += chunk
            while True:
                # Find the boundary in the buffer
                boundary_index = buffer.find(b'--' + boundary.encode())
                if boundary_index == -1:
                    break
                
                # Find the end of the header section
                header_end_index = buffer.find(b'\r\n\r\n', boundary_index)
                if header_end_index == -1:
                    break
                
                # Find the next boundary to determine the end of the frame data
                next_boundary_index = buffer.find(b'--' + boundary.encode(), header_end_index + 4)
                if next_boundary_index == -1:
                    break
                
                # Extract the frame data
                frame_data = buffer[header_end_index + 4:next_boundary_index]
                process_frame(frame_data)
                
                # Remove the processed frame from the buffer
                buffer = buffer[next_boundary_index:]

if __name__ == '__main__':
    stream_url = 'http://localhost:8086/stream'
    stream_frames(stream_url)