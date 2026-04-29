import torch
import os

def save_video_clip_stream(video_path, save_path):
    # Load and preprocess video
    pixel_values = load_video(video_path).half().cuda().unsqueeze(0)
    
    # Extract features
    with torch.no_grad():
        video_features = video_encoder(pixel_values)
        last_hidden_state = video_features.last_hidden_state.squeeze(0)  # (T, F)
    
    # Save to stream format
    data = {
        "video_id": os.path.basename(video_path),
        "clip_features": last_hidden_state.cpu(),  # Save on CPU to avoid GPU memory usage
    }
    torch.save(data, save_path)
    print(f"Saved stream file: {save_path}")

# === Example usage ===
video_path = "../../Apollo/nodding.mp4"
save_path = "nodding.video.clip.stream"
save_video_clip_stream(video_path, save_path)

