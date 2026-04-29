import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def extract_middle_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open {video_path}")
        return None
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    middle_index = frame_count // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_index)

    success, frame = cap.read()
    cap.release()
    if success:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        print(f"Failed to read frame from {video_path}")
        return None

def find_videos(subfolder_path, label, max_count=2):
    videos = []
    for file in os.listdir(subfolder_path):
        if label in file and file.endswith('.mp4'):
            videos.append(os.path.join(subfolder_path, file))
            if len(videos) == max_count:
                break
    return videos

def process_split(split_dir):
    frames = []

    for subfolder in sorted(os.listdir(split_dir)):
        subfolder_path = os.path.join(split_dir, subfolder)
        if not os.path.isdir(subfolder_path):
            continue

        expert_videos = find_videos(subfolder_path, 'expert', max_count=2)
        novice_videos = find_videos(subfolder_path, 'novice', max_count=2)

        for e_vid, n_vid in zip(expert_videos, novice_videos):
            e_frame = extract_middle_frame(e_vid)
            n_frame = extract_middle_frame(n_vid)

            if e_frame is not None and n_frame is not None:
                e_frame = cv2.resize(e_frame, (320, 240))
                n_frame = cv2.resize(n_frame, (320, 240))
                combined = np.concatenate((e_frame, n_frame), axis=1)  # side-by-side: expert | novice
                frames.append(combined)

    return frames

# Process each split
base_dir = "/mnt/dhanujaw/Noxi/"
splits = ['train', 'test', 'val']
split_results = []

# Get max number of frames across splits to pad later
max_rows = 0

for split in splits:
    split_dir = os.path.join(base_dir, split)
    if os.path.exists(split_dir):
        print(f"Processing split: {split}")
        frames = process_split(split_dir)
        split_results.append(frames)
        max_rows = max(max_rows, len(frames))
    else:
        split_results.append([])
        print(f"Directory not found: {split_dir}")

# Pad each split with white images to equal number of rows
for i in range(len(split_results)):
    num_to_pad = max_rows - len(split_results[i])
    if num_to_pad > 0:
        pad_image = np.ones((240, 640, 3), dtype=np.uint8) * 255
        split_results[i].extend([pad_image] * num_to_pad)

# Now stack each split vertically, then concatenate horizontally
column_images = [np.concatenate(frames, axis=0) for frames in split_results]
final_image = np.concatenate(column_images, axis=1)

# Save and show
save_path = "./all_splits_side_by_side.jpg"
cv2.imwrite(save_path, cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))
print(f"Saved final side-by-side image: {save_path}")

plt.imshow(final_image)
plt.title("Train | Test | Val — Expert vs Novice")
plt.axis('off')
plt.show()
