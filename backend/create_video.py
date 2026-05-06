import cv2
import os

def save_frames(video_path, output_folder, step=1):
    """
    Save frames from a video.
    
    Args:
        video_path (str): Path to input video file.
        output_folder (str): Directory where frames will be saved.
        step (int): Save every 'step' frame (e.g., step=5 → saves every 5th frame).
    """
    # Create output folder if not exists
    os.makedirs(output_folder, exist_ok=True)

    # Load video
    cap = cv2.VideoCapture(video_path)

    frame_id = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Save only every `step`-th frame
        if frame_id % step == 0:
            filename = os.path.join(output_folder, f"frame_{frame_id:05d}.jpg")
            cv2.imwrite(filename, frame)
            saved += 1

        frame_id += 1

    cap.release()
    print(f"Done! Saved {saved} frames to: {output_folder}")

# Example usage
save_frames("expert_video_1131.mp4", "saved_frames", step=1)

