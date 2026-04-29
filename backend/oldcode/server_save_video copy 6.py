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

        # ✅ small safety (avoid wrong gesture)
        allowed_gestures = ["None","Waving","Nod","Thinking","Agreeing","Acknowledging"]
        if data.get("mixamo_gesture") not in allowed_gestures:
            data["mixamo_gesture"] = "None"

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
            text = "[no speech detected]"

        print("Transcription:", text)

        # -----------------------------
        # FAST VLM PROMPT (CSV)
        # -----------------------------
        question = f"""
You are a multimodal AI assistant for the SELA system.

Analyze the given video together with this user utterance:
{text}

Return ONLY CSV with EXACTLY 2 lines:
1) header
2) data

Columns (strict order):
utterance,age_range,gender,emotion,openness,conscientiousness,extraversion,agreeableness,neuroticism,language_or_cultural_cue,social_signals,agent_response,avatar_emotion,mixamo_gesture

Rules:
- text fields in quotes
- personality values 0-1
- emotion must match visible user emotion
- avatar_emotion must match emotion of the person
- mixamo_gesture must reflect visible hand gesture
- agent_response must be short natural sentence

Output ONLY CSV.
"""
#Allowed:
#emotion: "neutral","happy","angry","sad","fear","disgust","surprised"
#gesture: "None","Waving","Nod","Thinking","Agreeing","Acknowledging"

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