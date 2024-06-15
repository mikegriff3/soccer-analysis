import cv2
import os

def read_video(video_path):
  if not os.path.exists(video_path):
        raise FileNotFoundError(f"The input file {video_path} does not exist.")
  cap = cv2.VideoCapture(video_path)
  if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")
  frames = []
  while True:
    ret, frame = cap.read()
    if not ret:
      break
    frames.append(frame)
  
  cap.release()
  #print(f"Read {len(frames)} frames from the video")
  return frames

def save_video(frames, output_path, fps=24.0):
    if not frames:
        raise ValueError("No frames to save.")
    
    # Get frame size from the first frame
    frame_height, frame_width = frames[0].shape[:2]
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    if not out.isOpened():
        raise IOError(f"Cannot open video file for writing: {output_path}")
    
    for frame in frames:
        out.write(frame)
    
    out.release()