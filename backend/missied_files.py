import os
import csv
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import cv2

from modeling_internvl_model import InternVLChatModel
from transformers.utils import logging

# =========================
# Logger Settings
# =========================
logger = logging.get_logger()
logger.setLevel("INFO")

# =========================
# Model Settings
# =========================
model_path = 'OpenGVLab/InternVideo2_5_Chat_8B'
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# =========================
# Transform Function
# =========================
def build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

# =========================
# Video Loader
# =========================
def load_video(video_path, num_segments=8, input_size=448):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    transform = build_transform(input_size=input_size)
    frame_indices = np.linspace(0, max_frame, num_segments, dtype=int)

    pixel_values_list = []
    for frame_index in tqdm(frame_indices, desc="📄 Loading frames", leave=False):
        img = Image.fromarray(vr[frame_index].asnumpy()).convert("RGB")
        img = transform(img)
        pixel_values_list.append(img)

    pixel_values = torch.stack(pixel_values_list)
    return pixel_values

# =========================
# Save Video Clip Stream
# =========================
def save_video_clip_stream(save_path, last_hidden_state, video_id):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    data = {
        "video_id": video_id,
        "clip_features": last_hidden_state.cpu()
    }
    torch.save(data, save_path)

# =========================
# Main Evaluation Loop
# =========================
seg_root = '/mnt/dhanujaw/Noxi/seg_videos/'
stream_root = '/mnt/dhanujaw/Noxi/stream_videos_8fps_spatial/'
output_path = '/mnt/dhanujaw/Noxi/spartially_seg_videos/'
log_file_path = 'missing_videos.log'

missing_entries = []

for split in tqdm(['train', 'val', 'test'], desc="📂 Processing splits"):
    split_dir = os.path.join(seg_root, split)
    stream_root_dir = os.path.join(stream_root, split)

    if not os.path.exists(split_dir):
        continue

    os.makedirs(stream_root_dir, exist_ok=True)

    video_ids = os.listdir(split_dir)
    for video_id in tqdm(video_ids, desc=f"🎞️ {split} videos", leave=False):
        video_folder = os.path.join(split_dir, video_id)
        video_folder_fmap = os.path.join(stream_root_dir, video_id)

        if not os.path.isdir(video_folder):
            continue

        os.makedirs(video_folder_fmap, exist_ok=True)

        video_files = sorted([f for f in os.listdir(video_folder) if f.endswith('.mp4') and '_video_' in f])
        for file in tqdm(video_files, desc=f"▶️ {video_id}", leave=False):
            condition = file.split('_')[0]
            if condition == "novice":
                continue

            novice = f"novice_{file.split('_')[1]}_{file.split('_')[2]}"
            novice_path = os.path.join(video_folder, novice)
            expert_path = os.path.join(video_folder, file)

            if os.path.exists(novice_path) and os.path.exists(expert_path):
                continue  # both exist
            else:
                missing_entries.append({
                    "split": split,
                    "video_id": video_id,
                    "expert": file,
                    "novice": novice,
                    "missing_expert": not os.path.exists(expert_path),
                    "missing_novice": not os.path.exists(novice_path)
                })

# =========================
# Save Missing Logs
# =========================
with open(log_file_path, 'w') as f:
    f.write("Missing Video File Report\n")
    f.write("==========================\n")
    for entry in missing_entries:
        f.write(
            f"[{entry['split']}] {entry['video_id']}: "
            f"Missing: "
            f"{'Expert' if entry['missing_expert'] else ''}"
            f"{' & ' if entry['missing_expert'] and entry['missing_novice'] else ''}"
            f"{'Novice' if entry['missing_novice'] else ''} "
            f"(Expert: {entry['expert']} | Novice: {entry['novice']})\n"
        )

print(f"\n✅ Missing video report saved to: {log_file_path}")

