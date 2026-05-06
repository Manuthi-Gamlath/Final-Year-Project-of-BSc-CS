import cv2
import os

# Input and output paths
input_path = '/mnt/dhanujaw/Noxi/train/001/expert.video.mp4'
output_path = './expert_10s.mp4'

# Create a VideoCapture object
cap = cv2.VideoCapture(input_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error opening video file.")
    exit()

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
codec = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for output video

# Frame range: from 2nd (index 1) to 10th (index 9)
start_frame = 0
end_frame = 62
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("Total Frames:", total_frames, "FPS:", fps, "Width:", width, "Height:", height)

# Create VideoWriter for output
out = cv2.VideoWriter(output_path, codec, fps, (width, height))

frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if start_frame <= frame_idx <= end_frame:
        out.write(frame)

    frame_idx += 1
    if frame_idx > end_frame:
        break

# Release resources
cap.release()
out.release()
print(f"Saved video from 2nd to 10th frame to: {output_path}")
