#%%
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
import cv2
import os
import argparse


class VideoFrameExtractor:
    def __init__(self, media_dir, output_folder, seconds_per_frame):
        self.media_dir = Path(media_dir)
        self.output_folder = Path(output_folder)
        self.seconds_per_frame = seconds_per_frame

    def get_final_frame_count(self, video_path):
        video = cv2.VideoCapture(str(video_path))
        fps = video.get(cv2.CAP_PROP_FPS)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = int(fps * self.seconds_per_frame)
        
        total_extracted = total_frames // frame_interval
        video.release()
        return total_extracted

    def extract_frames_single_video(self, video_path):
        """Extract frames from a single video"""
        video_name = video_path.stem
        video_output_folder = self.output_folder / video_name
        os.makedirs(video_output_folder, exist_ok=True)
        
        video = cv2.VideoCapture(str(video_path))
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * self.seconds_per_frame)
        
        frame_count = 0
        saved_count = 0
        
        while True:
            success, frame = video.read()
            if not success:
                break
                
            if frame_count % frame_interval == 0:
                output_path = video_output_folder / f'frame_{saved_count:05d}.jpg'
                cv2.imwrite(str(output_path), frame)
                saved_count += 1
                
            frame_count += 1
        
        video.release()
        return f'Extracted {saved_count} frames from {video_name}'

    def extract_frames_parallel(self, num_processes=None):
        """Extract frames from multiple videos in parallel"""
        video_files = list(self.media_dir.glob('*.mp4'))
        
        with Pool(processes=num_processes) as pool:
            results = list(tqdm(
                pool.imap(self.extract_frames_single_video, video_files),
                total=len(video_files),
                desc='Processing videos'
            ))


def main():
    parser = argparse.ArgumentParser(description='Extract frames from videos')
    parser.add_argument('--media-dir', type=str, default='media',
                      help='Directory containing input videos')
    parser.add_argument('--output-dir', type=str, 
                      default='datasets/resized_media/mobile_images',
                      help='Output directory for extracted frames')
    parser.add_argument('--seconds-per-frame', type=float, default=20,
                      help='Seconds between extracted frames')

    args = parser.parse_args()

    extractor = VideoFrameExtractor(
        media_dir=args.media_dir,
        output_folder=args.output_dir,
        seconds_per_frame=args.seconds_per_frame
    )
    extractor.extract_frames_parallel()


if __name__ == '__main__':
    main()