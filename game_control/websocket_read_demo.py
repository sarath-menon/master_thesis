import websocket
import threading
import time
import io
from PIL import Image
import cv2
import numpy as np

frame_count = 0
start_time = time.time()

def on_message(ws, message):
    global frame_count, start_time
    
    # Assuming the server sends binary JPEG images
    image_stream = io.BytesIO(message)
    image = Image.open(image_stream)
    img_np = np.array(image)

    # Convert BGR to RGB
    img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

    # print(f"Image resolution: {img_np.shape[1]}x{img_np.shape[0]}")

    cv2.imshow('frame', img_np)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()


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
        # You can send messages to the server here if needed
        # ws.send("Hello Server")
        while True:
            time.sleep(20)  # Keep the connection open
    thread = threading.Thread(target=run)
    thread.daemon = True  # Set thread as daemon so it closes with the main program
    thread.start()

if __name__ == "__main__":
    websocket.enableTrace(False)
    ws = websocket.WebSocketApp("ws://localhost:8086/stream_socket",
                                on_open=on_open,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)

    ws.run_forever()

