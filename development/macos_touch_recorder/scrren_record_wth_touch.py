import os
import time
import subprocess
from pynput import mouse
from datetime import datetime

def start_screen_recording(x1=0, y1=0, x2=1920, y2=1080):
    # Calculate width and height from coordinates
    width = x2 - x1
    height = y2 - y1
    
    # Save to the root folder instead
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"screen_recording_{timestamp}.mov"
    
    # Build screencapture command
    cmd = [
        "screencapture",
        "-v",  # Enable video recording
        "-R", f"{x1},{y1},{width},{height}",  # Region coordinates
        output_file
    ]
    
    print(f"Saving recording to: {output_file}")
    return subprocess.Popen(cmd)

def stop_screen_recording():
    # Find and terminate screencapture process
    subprocess.run(["pkill", "screencapture"])

# Example usage - record a 500x500 region starting at (100,100)
start_time = datetime.now()
print(f"Recording started at: {start_time}")
start_screen_recording(100, 100, 600, 600)


# Wait for 10 seconds
time.sleep(10)

# Stop recording and listener
stop_screen_recording()
print(f"Recording stopped at: {datetime.now()}")