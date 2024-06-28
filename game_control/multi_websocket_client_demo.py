import websocket
import threading
import time
import io
from PIL import Image
import cv2
import numpy as np
import json
import queue

class WebSocketClient:
    def __init__(self):
        self.frame_count = 0
        self.start_time = time.time()
        self.image_queue = queue.LifoQueue(maxsize=100)

        # Example variables - replace these with actual values
        self.image_data = b'...'  # This should be your _imageByte data
        self.width = 1920      # Replace with actual image width
        self.height = 1080        # Replace with actual image height

    def main_thread(self):
        while True:
            if self.image_queue.qsize() > 0:
                image = self.image_queue.get()
                cv2.imshow('image', image)
                time.sleep(0.001)
                if cv2.waitKey(1) & 0xFF == ord('q'):  
                    break

    def on_image_message(self, ws, message):
        # Create an image from the byte data
        image = Image.frombytes(mode='RGBA', size=(self.width, self.height), data=message)

        # Convert the image to a numpy array only if necessary
        img_np = np.array(image)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

        if not self.image_queue.full():
            self.image_queue.put(img_np)
        else:
            self.image_queue.queue.clear()

        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        if elapsed_time > 1:  # Update FPS every second
            fps = self.frame_count / elapsed_time
            print(f"Receiver FPS: {fps:.2f}")
            self.frame_count = 0
            self.start_time = time.time()

    def on_keypress_open(self, ws):
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

    def on_image_open(self, ws):
        def run(*args):
            # You can send messages to the server here if needed
            # ws.send("Hello Server")
            pass

        thread = threading.Thread(target=run)
        thread.daemon = True  # Set thread as daemon so it closes with the main program
        thread.start()

    def on_keypress_message(self, ws, message):
        print(f"Keypress Received: {message}")

    def on_error(self, ws, error):
        print(f"Error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        print("### closed ###")

    def start(self):
        websocket.enableTrace(False)
        
        # Create a separate WebSocket for receiving images
        ws_image = websocket.WebSocketApp("ws://localhost:8086/stream_websocket",                     
                                           on_message=self.on_image_message,
                                           on_open=self.on_image_open,
                                           on_error=self.on_error,
                                           on_close=self.on_close)
        
        # Create a separate WebSocket for sending keypress events
        ws_keypress = websocket.WebSocketApp("ws://localhost:8086/keypress_websocket",
                                              on_open=self.on_keypress_open,
                                              on_message=self.on_keypress_message,
                                              on_error=self.on_error,
                                              on_close=self.on_close)

        try:
            image_thread = threading.Thread(target=ws_image.run_forever)
            keypress_thread = threading.Thread(target=ws_keypress.run_forever)
            image_thread.start()
            keypress_thread.start()
            self.main_thread()
        except KeyboardInterrupt:
            print("Interrupted by user, stopping...")
            ws_image.close()
            ws_keypress.close()

if __name__ == "__main__":
    client = WebSocketClient()
    client.start()