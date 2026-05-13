# media_utils.py
import os
import subprocess
from pathlib import Path

def _run(cmd: list[str]) -> None:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\nCMD: {' '.join(cmd)}\nSTDERR:\n{p.stderr}")

def extract_muted_video(input_video: str, out_dir: str) -> str:
    """
    Create a video-only file (no audio stream).
    Uses stream copy when possible (fast).
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    base = Path(input_video).stem
    out_path = os.path.join(out_dir, f"{base}_noaudio.mp4")

    cmd = [
        "ffmpeg", "-y",
        "-i", input_video,
        "-an",            # drop audio
        "-c:v", "copy",   # copy video if possible
        out_path
    ]
    _run(cmd)
    return out_path

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
