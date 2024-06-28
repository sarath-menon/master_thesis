import websocket
import threading
import time
import io
from PIL import Image
import cv2
import numpy as np
import json

# Example variables - replace these with actual values
image_data = b'...'  # This should be your _imageByte data
width = 1920      # Replace with actual image width
height = 1080        # Replace with actual image height

def main(ws, n_frames):

    global frame_count, start_time

    message = {"action": "request_frames", "count": n_frames}
    ws.send(json.dumps(message))

    for i in range(n_frames):
        message = ws.recv()

        # Create an image from the byte data
        image = Image.frombytes(mode='RGBA', size=(width, height), data=message)

        # Convert the image to a numpy array only if necessary
        img_np = np.array(image)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

        # Show the image using cv2, consider reducing frequency of imshow if possible
        cv2.imshow('Image', img_np)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Allow for a quit option
            cv2.destroyAllWindows()
            raise SystemExit("User requested exit.")
        if cv2.getWindowProperty('Image', cv2.WND_PROP_VISIBLE) < 1:
            break

if __name__ == "__main__":
    websocket.enableTrace(False)
    ws = websocket.WebSocket()
    ws.connect("ws://localhost:8086/stream_websocket")

    n_frames = 100

    main(ws, n_frames)

