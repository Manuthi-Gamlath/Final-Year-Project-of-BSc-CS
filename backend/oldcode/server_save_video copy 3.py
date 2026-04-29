# app.py
#uvicorn server_save_video:app --host 0.0.0.0 --port 8891
import os
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

from media_utils import extract_muted_video, extract_audio
from model_service import load_model_once, infer_video, is_model_loaded
from faster_whisper import WhisperModel
model = WhisperModel("base", device="cpu", compute_type="int8")
def transcribe_audio(audio_path):
    # Use "base" or "small" for faster real-time
    

    segments, info = model.transcribe(audio_path)

    full_text = ""
    for segment in segments:
        full_text += segment.text + " "

    return full_text.strip()



# -----------------------------
# CONFIG
# -----------------------------
SAVE_DIR = "uploads"
DERIV_DIR = os.path.join(SAVE_DIR, "derived")
Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
Path(DERIV_DIR).mkdir(parents=True, exist_ok=True)

app = FastAPI()

# ✅ CORS so your laptop HTML can call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # for testing; later set to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# STARTUP: LOAD MODEL ONCE
# -----------------------------
@app.on_event("startup")
def _startup():
    # ✅ model loads ONCE here
    load_model_once()


# -----------------------------
# ROUTES
# -----------------------------
@app.get("/health")
def health():
    return {
        "ok": True,
        "message": "backend is reachable",
        "model_loaded": is_model_loaded()
    }


@app.post("/upload")
async def upload_video(
    file: UploadFile = File(...),
    seconds: str = Form(None),
    user_id: str = Form(None),
    #question: str = Form("""You are SELA (Socially Aware Embodied Language Agent).  User Speech Transcript: "hi how are you?"  Analyze the provided video and determine internally: - The person’s facial expression (primary emotion and intensity) - The person’s gender - The person’s estimated age - The person’s personality traits using the Big Five model (Openness, Extraversion, Agreeableness, Neuroticism) - The person’s hand gestures and engagement level  Then, based on: 1. What the user said, and 2. The detected visual and personality attributes,  Generate a natural conversational message to the person.  The message should: - Reflect the emotional tone of what they said. - Align with their facial expression and intensity. - Adapt tone according to personality traits. - Be age-appropriate. - Match their engagement level. - Be empathetic and socially intelligent. - Ask one meaningful follow-up question. - Be under 80 words.  Do NOT mention the analysis. Do NOT list attributes. Output ONLY the final conversational message directed to the person
#""")
    question: str = Form("""You are SELA (Socially Aware Embodied Language Agent). User Speech Transcript: "{text}". Analyze the provided video and determine internally the person’s facial expression (primary emotion and intensity), gender, estimated age, personality traits using the Big Five model (Openness, Extraversion, Agreeableness, Neuroticism), and hand gestures with engagement level. Then, based on what the user said and the detected visual and personality attributes, generate a natural conversational message to the person. The message should reflect the emotional tone of what they said, align with their facial expression and intensity, adapt tone according to personality traits, be age-appropriate, match engagement level, be empathetic and socially intelligent, ask one meaningful follow-up question, and be under 80 words. Do NOT mention the analysis. Do NOT list attributes. Output ONLY the final conversational message directed to the person."""
                         
    )
):
   
    #ts = datetime.now().strftime("%Y%m%d_%H%M%S")


    safe_user = (user_id or "user").replace("/", "_").replace("\\", "_")
    safe_name = (file.filename or "upload.mp4").replace("/", "_").replace("\\", "_")

    filename = f"{safe_user}_{safe_name}"
    saved_path = os.path.join(SAVE_DIR, filename)

    # ✅ Save uploaded file
    data = await file.read()
    with open(saved_path, "wb") as f:
        f.write(data)

    size_kb = len(data) / 1024.0

    # ✅ Extract muted video + audio
    muted_video_path = extract_muted_video(saved_path, DERIV_DIR)
    audio_path = extract_audio(saved_path, DERIV_DIR, sr=16000)
    audio_file_path = "uploads/derived/dhanuja_recording_5s_audio.wav"
    text = transcribe_audio(audio_file_path)
    print("Transcription isss:", text)
    user_utterance=text
    question = f"""You are a multimodal AI assistant for the SELA system. Analyze the given video together with this user utterance: {repr(user_utterance)}. Return the result as a valid CSV table with exactly 2 lines only: (1) one header row and (2) one data row. Do not output key-value pairs. Do not put one field per line. Use exactly these columns in this exact order: utterance,age_range,gender,emotion,openness,conscientiousness,extraversion,agreeableness,neuroticism,language_or_cultural_cue,social_signals,agent_response,avatar_emotion,mixamo_gesture. The fields utterance, age_range, gender, emotion, language_or_cultural_cue, social_signals, agent_response, avatar_emotion, and mixamo_gesture must be text in double quotes. The fields openness, conscientiousness, extraversion, agreeableness, and neuroticism must be numeric values between 0 and 1. social_signals must be one single quoted field containing 3 to 5 short comma-separated signals. agent_response must be one short natural sentence. avatar_emotion must be suitable for the avatar face. mixamo_gesture must be a valid short gesture label such as "Idle", "Talking", "Nod", or "Waving". Output only the CSV header and one CSV data row, nothing else."""
  

    # ✅ Run inference using already-loaded model (no reload)
    output = infer_video(
        video_path=muted_video_path,
        audio_path=audio_path,
        question=question
    )
    print(output)
    return {
        "ok": True,
        "message": output,
        "saved_path": saved_path,
    }