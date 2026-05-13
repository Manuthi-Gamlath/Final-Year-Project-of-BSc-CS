import subprocess

input_audio = "./internvideo2.5_code/uploads/derived/dhanuja_recording_5s_audio.wav"
output_video = "video_dha.mp4"

# FFmpeg command: create black video + add audio
cmd = [
    "ffmpeg",
    "-y",  # overwrite output
    "-f", "lavfi",
    "-i", "color=c=black:s=1280x720:r=25",  # black video
    "-i", input_audio,  # input audio
    "-shortest",  # match duration to audio
    "-c:v", "libx264",
    "-c:a", "aac",
    output_video
]

subprocess.run(cmd, check=True)

print("Video created:", output_video)