from faster_whisper import WhisperModel
import time

# Load model once (good practice)
model = WhisperModel("base", device="cpu", compute_type="int8")

def transcribe_audio(audio_path):
    start_time = time.time()  # ⏱️ start timer

    segments, info = model.transcribe(audio_path)

    full_text = ""
    for segment in segments:
        full_text += segment.text + " "

    end_time = time.time()  # ⏱️ end timer

    inference_time = end_time - start_time

    return full_text.strip(), inference_time


if __name__ == "__main__":
    audio_file_path = "uploads/derived/dhanuja_recording_5s_audio.wav"

    text, latency = transcribe_audio(audio_file_path)

    print("Transcription:", text)
    print(f"Inference Time: {latency:.4f} seconds")