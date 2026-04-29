# app.py
# Run: uvicorn app:app --host 0.0.0.0 --port 8891

import os
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel

from media_utils import extract_muted_video, extract_audio
from model_service import load_model_once, infer_video, is_model_loaded


# -----------------------------
# CONFIG
# -----------------------------
SAVE_DIR = "uploads"
DERIV_DIR = os.path.join(SAVE_DIR, "derived")

Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
Path(DERIV_DIR).mkdir(parents=True, exist_ok=True)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Whisper (fast)
whisper_model = WhisperModel("base", device="cpu", compute_type="int8")


# -----------------------------
# HELPERS
# -----------------------------
def transcribe_audio(audio_path: str):
    segments, _ = whisper_model.transcribe(audio_path)
    return " ".join(s.text.strip() for s in segments if s.text.strip()).strip()


# ✅ ADDED: simple keyword classification
def classify_emotion(text):
    t = str(text).lower()

    if any(k in t for k in ["happy", "joy", "smile", "cheerful"]):
        return "happy"
    if any(k in t for k in ["angry", "mad", "furious", "annoy"]):
        return "angry"
    if any(k in t for k in ["sad", "cry", "unhappy", "down"]):
        return "sad"
    if any(k in t for k in ["fear", "scared", "afraid", "nervous"]):
        return "fear"
    if any(k in t for k in ["disgust", "gross", "repulse"]):
        return "disgust"
    if any(k in t for k in ["surprise", "shocked", "amazed"]):
        return "surprised"

    return "neutral"


def classify_gesture(text):
    t = str(text).lower()

    if any(k in t for k in ["wave", "waving", "hello", "greet"]):
        return "Waving"
    if any(k in t for k in ["nod", "nodding"]):
        return "Nod"
    if any(k in t for k in ["think", "thinking", "ponder"]):
        return "Thinking"
    if any(k in t for k in ["agree", "yes", "approval"]):
        return "Agreeing"
    if any(k in t for k in ["acknowledge", "ok", "okay", "understand", "listen"]):
        return "Acknowledging"

    return "None"


def parse_csv_response(text: str):
    """
    Fast CSV → JSON parser
    """
    try:
        lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
        if len(lines) < 2:
            return {"error": "invalid format", "raw": text}

        headers = [h.strip() for h in lines[0].split(",")]
        values = [v.strip().strip('"') for v in lines[1].split(",")]

        data = {}
        for h, v in zip(headers, values):
            if h in ["openness","conscientiousness","extraversion","agreeableness","neuroticism"]:
                try:
                    data[h] = float(v)
                except:
                    data[h] = 0.5
            else:
                data[h] = v

        # ✅ ONLY CHANGE: classify outputs
        data["avatar_emotion"] = classify_emotion(data.get("avatar_emotion", ""))
        data["mixamo_gesture"] = classify_gesture(data.get("mixamo_gesture", ""))

        return data

    except Exception as e:
        return {"error": str(e), "raw": text}


# -----------------------------
# STARTUP
# -----------------------------
@app.on_event("startup")
def startup():
    load_model_once()


# -----------------------------
# ROUTES
# -----------------------------
@app.get("/health")
def health():
    return {
        "ok": True,
        "model_loaded": is_model_loaded()
    }


@app.post("/upload")
async def upload_video(
    file: UploadFile = File(...),
    user_id: str = Form(None),
):

    try:
        # -----------------------------
        # SAVE FILE
        # -----------------------------
        safe_user = (user_id or "user").replace("/", "_")
        safe_name = (file.filename or "upload.mp4").replace("/", "_")
        filename = f"{safe_user}_{safe_name}"
        saved_path = os.path.join(SAVE_DIR, filename)

        data = await file.read()
        with open(saved_path, "wb") as f:
            f.write(data)

        # -----------------------------
        # EXTRACT MEDIA
        # -----------------------------
        muted_video_path = extract_muted_video(saved_path, DERIV_DIR)
        audio_path = extract_audio(saved_path, DERIV_DIR, sr=16000)

        # -----------------------------
        # TRANSCRIBE
        # -----------------------------
        text = transcribe_audio(audio_path)
        if not text:
            text = ""
        text="I just got an A in my project!"
        print("Transcription:", text)

        # -----------------------------
        # FAST VLM PROMPT (CSV)
        # -----------------------------
        question = f"""
Analyze the video and the user utterance.

User utterance: "{text}"

Return ONLY JSON.

Tasks:
- detect emotion of the person from: neutral, happy, sad, angry, fear, disgust, surprised
- detect gesture of the person from: Waving, Nod, Thinking, Agreeing, Acknowledging, Clap
- ethnicity is from: Sri lankan, chineese, black
- estimate lightweight personality scores from 0.00 to 1.00
- generate one short empathetic reply
- select matching avatar_emotion
- mixamo_gesture is the repose to what person does 

JSON schema:
{{
  "utterance": "{text}",
  "age_range": "child|teen|young_adult|adult|older_adult|unknown",
  "gender": "male|female|unknown",
  "emotion": "neutral|happy|sad|angry|fear|disgust|surprised",
  "openness": 0-1,
  "conscientiousness": 0-1,
  "extraversion": 0-1,
  "agreeableness": 0-1,
  "neuroticism": 0-1,
  "ethnicity":  chineese| black |Sri lankan,
  "agent_response": "short empathetic reply",
  "avatar_emotion": "neutral|happy|sad|angry|fear|disgust|surprised",
  "mixamo_gesture": 
}}

Rules:
- JSON only
- no markdown
- no extra text
- emotion must reflect the visible user emotion
- agent_response must be socially aware
- avatar_emotion must align with the detected emotion
"""

        # -----------------------------
        # INFERENCE
        # -----------------------------
        output = infer_video(
            video_path=muted_video_path,
            audio_path=audio_path,
            question=question
        )

        print("Raw output:", output)

        # -----------------------------
        # PARSE (FAST)
        # -----------------------------
        parsed = parse_csv_response(output)

        # -----------------------------
        # RESPONSE
        # -----------------------------
        return {
            "ok": True,
            "message": parsed,
            "raw": output,
            "transcription": text,
            "saved_path": saved_path
        }

    except Exception as e:
        return {
            "ok": False,
            "error": str(e)
        }