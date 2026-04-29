# app.py
import os
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

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
    question: str = Form(
        """You are SELA (Socially Aware Embodied Language Agent), a real-time multimodal conversational agent.

You are given:

1. User Speech Transcript:
"{transcribed_text}"

2. Detected Facial Emotion:
Primary Emotion: {face_emotion}
Valence: {valence_score}
Arousal: {arousal_score}

3. Detected Vocal Emotion:
Tone: {voice_emotion}
Energy Level: {energy_level}

4. Inferred Personality Cues (lightweight estimation):
{personality_traits}

5. Conversation History:
{previous_context}

Your task:
- Understand the user's emotional and psychological state.
- Adapt tone, empathy level, and word choice accordingly.
- Provide a socially intelligent, emotionally aligned response.
- Keep response natural and under 80 words.
- Maintain conversational continuity.
- Do NOT mention the analysis above.
- Respond as a caring embodied conversational agent.

Output format:
Response: <text response>
Avatar Emotion: <emotion label for avatar animation>
Speech Style: <calm / energetic / supportive / neutral>
"""
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

    # ✅ Run inference using already-loaded model (no reload)
    output = infer_video(
        video_path=muted_video_path,
        audio_path=audio_path,
        question=question
    )

    return {
        "ok": True,
        "message": output,
        "saved_path": saved_path,
    }