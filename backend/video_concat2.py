import cv2
import os

# Paths
expert_path = '/data/Noxi/seg_videos/train/001/expert_video_100.mp4'
novice_path = '/data/Noxi/seg_videos/train/001/novice_video_100.mp4'
output_path = './spatial_concat_video.mp4'

# Open both videos
cap_expert = cv2.VideoCapture(expert_path)
cap_novice = cv2.VideoCapture(novice_path)

# Check if both videos opened
if not cap_expert.isOpened() or not cap_novice.isOpened():
    raise IOError("Could not open one of the videos")

# Get properties from both videos
fps = int(cap_expert.get(cv2.CAP_PROP_FPS))
width = int(cap_expert.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap_expert.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Check if both videos have the same height, resize if necessary
novice_width = int(cap_novice.get(cv2.CAP_PROP_FRAME_WIDTH))
novice_height = int(cap_novice.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Output frame width is sum of both widths, height is max of both
out_width = width + novice_width
out_height = max(height, novice_height)

# Define the video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))

while True:
    ret1, frame1 = cap_expert.read()
    ret2, frame2 = cap_novice.read()

    if not ret1 or not ret2:
        break

    # Resize frames to the same height if needed
    if frame1.shape[0] != frame2.shape[0]:
        common_height = min(frame1.shape[0], frame2.shape[0])
        frame1 = cv2.resize(frame1, (int(frame1.shape[1] * common_height / frame1.shape[0]), common_height))
        frame2 = cv2.resize(frame2, (int(frame2.shape[1] * common_height / frame2.shape[0]), common_height))

    # Horizontally stack the frames
    combined_frame = cv2.hconcat([frame1, frame2])

    # Write the combined frame
    out.write(combined_frame)

# Release everything
cap_expert.release()
cap_novice.release()
out.release()

print(f"Spatial concatenation saved to: {output_path}")
