# media_utils.py
import os
import subprocess
from pathlib import Path
import os
import cv2
import numpy as np

def _run(cmd: list[str]) -> None:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\nCMD: {' '.join(cmd)}\nSTDERR:\n{p.stderr}")





#def extract_uniform_sampled_video_opencv(input_path: str, out_dir: str, n_frames: int) -> str:
def extract_uniform_sampled_video_opencv(input_path: str, out_dir: str, n_frames: int) -> str:
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 10.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = []

    # Fallback: read all frames if frame count is unavailable
    if total_frames <= 0:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)

        cap.release()

        total_frames = len(frames)
        if total_frames == 0:
            raise RuntimeError(f"Could not read any frames from video: {input_path}")

        height, width = frames[0].shape[:2]
        n_frames = max(1, min(n_frames, total_frames))
        frame_indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)

        base = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(out_dir, f"{base}_sampled_{n_frames}.mp4")

        out_fps = max(1.0, min(float(fps), float(n_frames)))
        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            out_fps,
            (width, height)
        )

        for idx in frame_indices:
            writer.write(frames[int(idx)])

        writer.release()
        return output_path

    # Normal path when frame count is available
    n_frames = max(1, min(n_frames, total_frames))
    frame_indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)

    base = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(out_dir, f"{base}_sampled_{n_frames}.mp4")

    out_fps = max(1.0, min(float(fps), float(n_frames)))
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        out_fps,
        (width, height)
    )

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            writer.write(frame)

    cap.release()
    writer.release()

    return output_path

def extract_muted_video(input_path: str, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(out_dir, f"{base}_noaudio.mp4")

    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-an",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed:\nCMD: {' '.join(cmd)}\nSTDERR:\n{result.stderr}"
        )

    return output_path

def extract_audio(input_video: str, out_dir: str, sr: int = 16000) -> str:
    """
    Extract audio track to .wav (PCM) for ML pipelines.
    Output: <stem>_audio.wav

    sr: target sampling rate (default 16000)
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    base = Path(input_video).stem
    out_path = os.path.join(out_dir, f"{base}_audio.wav")

    cmd = [
        "ffmpeg", "-y",
        "-i", input_video,
        "-vn",                 # no video
        "-ac", "1",            # mono
        "-ar", str(sr),        # sample rate
        "-c:a", "pcm_s16le",   # wav PCM
        out_path
    ]
    _run(cmd)
    return out_path
