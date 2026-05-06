import cv2
import os

# Paths
expert_path = '/data/Noxi/seg_videos/train/001/expert_video_100.mp4'
novice_path = '/data/Noxi/seg_videos/train/001/novice_video_100.mp4'
output_path = './temporal_concat_video.mp4'

# Open both videos
cap_expert = cv2.VideoCapture(expert_path)
cap_novice = cv2.VideoCapture(novice_path)

# Check if both videos opened
if not cap_expert.isOpened() or not cap_novice.isOpened():
    raise IOError("Could not open one of the videos")

# Get properties from the first video
fps = int(cap_expert.get(cv2.CAP_PROP_FPS))
width = int(cap_expert.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap_expert.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

def write_video_frames(cap, out, label):
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

# Write expert video first
write_video_frames(cap_expert, out, "Expert")

# Then novice video
write_video_frames(cap_novice, out, "Novice")

# Release everything
cap_expert.release()
cap_novice.release()
out.release()

print(f"Temporal concatenation saved to: {output_path}")
