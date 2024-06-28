import websocket
import threading
import time
import io
from PIL import Image
import cv2
import numpy as np
import json
import queue

image_queue = queue.Queue(maxsize=100)

frame_count = 0
start_time = time.time()

# Example variables - replace these with actual values
image_data = b'...'  # This should be your _imageByte data
width = 1920      # Replace with actual image width
height = 1080        # Replace with actual image height

def display_images():
    while True:
        if not image_queue.empty():
            img_np = image_queue.get()
            cv2.imshow('Image', img_np)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                raise SystemExit("User requested exit.")

def on_image_message(ws, message):
    global frame_count, start_time

    # Create an image from the byte data
    image = Image.frombytes(mode='RGBA', size=(width, height), data=message)

    # Convert the image to a numpy array only if necessary
    img_np = np.array(image)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time > 1:  # Update FPS every second
        fps = frame_count / elapsed_time
        print(f"FPS: {fps:.2f}")
        frame_count = 0
        start_time = time.time()

    if frame_count % 10 == 0:
        cv2.imwrite(f'image_{int(time.time())}.png', img_np)

def on_keypress_open(ws):
    def run(*args):
        while True:
            request_data = {
                "action": "keypress",
                "key": "W",  # Example key
                "duration": "300"  # Duration in milliseconds
            }
            ws.send(json.dumps(request_data))
            print(f"Sent: {request_data}")

            time.sleep(5)  # Adjust the sleep time as needed
    thread = threading.Thread(target=run)
    thread.daemon = True
    thread.start()

def on_image_open(ws):
    def run(*args):
        # You can send messages to the server here if needed
        # ws.send("Hello Server")
        while True:
            time.sleep(20)  # Keep the connection open
    thread = threading.Thread(target=run)
    thread.daemon = True  # Set thread as daemon so it closes with the main program
    thread.start()

def on_keypress_message(ws, message):
    print(f"Keypress Received: {message}")

def on_error(ws, error):
    print(f"Error: {error}")

def on_close(ws, close_status_code, close_msg):
    print("### closed ###")

if __name__ == "__main__":
    websocket.enableTrace(False)
    
    # Create a separate WebSocket for receiving images
    ws_image = websocket.WebSocketApp("ws://localhost:8086/stream_websocket",                     
                                       on_message=on_image_message,
                                       on_open=on_image_open,
                                       on_error=on_error,
                                       on_close=on_close)
    
    # Create a separate WebSocket for sending keypress events
    ws_keypress = websocket.WebSocketApp("ws://localhost:8086/keypress_websocket",
                                          on_open=on_keypress_open,
                                          on_message=on_keypress_message,
                                          on_error=on_error,
                                          on_close=on_close)


    try:
        threading.Thread(target=ws_image.run_forever).start()
        threading.Thread(target=ws_keypress.run_forever).start()
        # display_images()
    except KeyboardInterrupt:
        print("Interrupted by user, stopping...")
        ws_image.close()
        ws_keypress.close()