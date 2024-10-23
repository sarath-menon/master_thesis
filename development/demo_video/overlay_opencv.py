#%%
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
from moviepy.video.fx.all import crop
from PIL import Image, ImageDraw, ImageFont
import numpy as np

#%%
import cv2
from PIL import Image, ImageDraw, ImageFont

def get_dynamic_text(frame_number, fps):
    # Calculate timestamp in seconds
    timestamp = frame_number / fps
    return f"Time: {timestamp:.2f}s"

def overlay_text_on_video(input_video_path, output_video_path, logo_path):
    # Define padding and positions
    padding = 20
    box_start_x = 100
    box_start_y = 50
    
    # Open the video file
    cap = cv2.VideoCapture(input_video_path)
    # Change the codec to mp4v
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    # Load a font
    font = ImageFont.truetype("./development/models/Poppins-Regular.ttf", 32)

    # Load the logo once outside the loop
    logo = Image.open(logo_path)
    
    # Calculate new dimensions maintaining aspect ratio
    target_height = 80  # Set your desired height
    aspect_ratio = logo.width / logo.height
    new_width = int(target_height * aspect_ratio)
    logo = logo.resize((new_width, target_height), Image.Resampling.LANCZOS)
    
    # Adjust fixed box width if needed based on new logo size
    fixed_box_width = max(300, new_width + 2 * padding)  # Ensure box is wide enough for logo
    fixed_box_height = 200

    frame_number = 0
    fps = cap.get(cv2.CAP_PROP_FPS)

    
    # Define the background box position with fixed dimensions
    bg_position = (
        box_start_x,
        box_start_y,
        box_start_x + fixed_box_width,
        box_start_y + fixed_box_height
    )
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to a PIL image
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        text = get_dynamic_text(frame_number, fps)
        
        # Calculate text size
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # Calculate centered positions within the fixed box
        logo_x = box_start_x + (fixed_box_width - logo.width) // 2
        logo_y = box_start_y + padding
        
        text_x = box_start_x + (fixed_box_width - text_width) // 2
        text_y = logo_y + logo.height + padding
        
        # Draw rounded rectangle background with fixed size
        draw.rounded_rectangle(
            bg_position,
            radius=15,
            fill=(18,18,26),
            outline=None
        )
        
        # Paste logo inside the black box
        pil_img.paste(logo, (logo_x, logo_y), mask=logo if logo.mode == 'RGBA' else None)
        
        # Draw text below the logo
        text_color = (255, 255, 255)  # white color
        draw.text((text_x, text_y), text, font=font, fill=text_color)

        # Convert the PIL image back to a frame
        frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # Write the frame to the output video
        out.write(frame)
        frame_number += 1

    cap.release()
    out.release()

#%%
logo_path = "./development/demo_video/nunu_logo_full.jpg"

overlay_text_on_video("./development/demo_video/selv.mp4", "./development/demo_video/output_video.mp4", logo_path)
# %%
