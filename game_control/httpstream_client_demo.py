import requests
from PIL import Image
from io import BytesIO
import cv2
import numpy as np

def process_frame(frame_data):
    Image.open(BytesIO(frame_data)).show()

def stream_frames(url):
    with requests.get(url, stream=True) as response:
        if response.status_code != 200:
            print("Failed to connect to the server.")
            return

        print(response.headers['X-Timestamp'])

        boundary = response.headers['content-type'].split('=')[1]
        buffer = b''
        
        for chunk in response.iter_content(chunk_size=1024):
            buffer += chunk
            while True:
                boundary_bytes = b'--' + boundary.encode()
                boundary_index = buffer.find(boundary_bytes)
                if boundary_index == -1:
                    break
                
                header_end_index = buffer.find(b'\r\n\r\n', boundary_index)
                if header_end_index == -1:
                    break
                
                next_boundary_index = buffer.find(boundary_bytes, header_end_index + 4)
                if next_boundary_index == -1:
                    break
                
                frame_data = buffer[header_end_index + 4:next_boundary_index]
                buffer = buffer[next_boundary_index:]  # Move the buffer pointer to the next part
                
                img_np = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
                cv2.imshow('frame', img_np)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                buffer = buffer[next_boundary_index:]

if __name__ == '__main__':
    stream_frames('http://localhost:8086/stream')