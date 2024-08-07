import websocket
import threading
import time
import io
from PIL import Image
import cv2
import numpy as np
import json

frame_count = 0
start_time = time.time()

# Example variables - replace these with actual values
image_data = b'...'  # This should be your _imageByte data
width = 1920      # Replace with actual image width
height = 1080        # Replace with actual image height

def on_message(ws, message):
    global frame_count, start_time

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

    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time > 1:  # Update FPS every second
        fps = frame_count / elapsed_time
        print(f"FPS: {fps:.2f}")
        frame_count = 0
        start_time = time.time()

def on_error(ws, error):
    print(f"Error: {error}")

def on_close(ws, close_status_code, close_msg):
    print("### closed ###")

def on_open(ws):
    def run(*args):
        message = {"action": "request_frames", "count": 100}
        ws.send(json.dumps(message))

    thread = threading.Thread(target=run)
    thread.daemon = True  # Set thread as daemon so it closes with the main program
    thread.start()

if __name__ == "__main__":
    websocket.enableTrace(False)
    ws = websocket.WebSocketApp("ws://localhost:8086/stream_websocket",
                                on_open=on_open,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)

    ws.run_forever()

