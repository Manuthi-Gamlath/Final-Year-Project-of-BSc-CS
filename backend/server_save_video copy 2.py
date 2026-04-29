# app.py
# Run: uvicorn app:app --host 0.0.0.0 --port 8891

import os
import uuid
import json
import re
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect
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
    if any(k in t for k in ["clap"]):
        return "Clap"

    return "None"


def normalize_server_message(data: dict):
    float_fields = [
        "openness",
        "conscientiousness",
        "extraversion",
        "agreeableness",
        "neuroticism",
    ]

    for field in float_fields:
        try:
            data[field] = float(data.get(field, 0.5))
        except Exception:
            data[field] = 0.5

    data["utterance"] = str(data.get("utterance", "") or "")
    data["age_range"] = str(data.get("age_range", "unknown") or "unknown")
    data["gender"] = str(data.get("gender", "unknown") or "unknown")
    data["emotion"] = classify_emotion(data.get("emotion", ""))
    data["ethnicity"] = str(data.get("ethnicity", "unknown") or "unknown")
    data["agent_response"] = str(
        data.get("agent_response", "Hello! How can I help you?") or "Hello! How can I help you?"
    )
    data["avatar_emotion"] = classify_emotion(
        data.get("avatar_emotion", data.get("emotion", ""))
    )
    data["mixamo_gesture"] = classify_gesture(data.get("mixamo_gesture", ""))

    return data


def parse_json_response(text: str):
    try:
        raw = str(text).strip()
        if not raw:
            return {"error": "empty model output", "raw": text}

        # case 1: pure JSON
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return normalize_server_message(parsed)
        except Exception:
            pass

        # case 2: JSON object inside extra text
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, dict):
                    return normalize_server_message(parsed)
            except Exception:
                pass

        return {"error": "invalid json format", "raw": text}

    except Exception as e:
        return {"error": str(e), "raw": text}


def build_question(text: str):
    return f"""
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


async def process_saved_video(saved_path: str, user_id: str = "user"):
    muted_video_path = extract_muted_video(saved_path, DERIV_DIR)
    audio_path = extract_audio(saved_path, DERIV_DIR, sr=16000)

    text = transcribe_audio(audio_path)
    if not text:
        text = ""

    print("Transcription:", text)

    question = build_question(text)

    output = infer_video(
        video_path=muted_video_path,
        audio_path=audio_path,
        question=question
    )

    print("Raw output:", output)

    parsed = parse_json_response(output)

    return {
        "ok": True,
        "message": parsed,
        "raw": output,
        "transcription": text,
        "saved_path": saved_path
    }


# -----------------------------
# STARTUP
# -----------------------------
@app.on_event("startup")
def startup():
    load_model_once(True)


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
        safe_user = (user_id or "user").replace("/", "_")
        safe_name = (file.filename or "upload.mp4").replace("/", "_")
        filename = f"{safe_user}_{safe_name}"
        saved_path = os.path.join(SAVE_DIR, filename)

        data = await file.read()
        with open(saved_path, "wb") as f:
            f.write(data)

        return await process_saved_video(saved_path, user_id=safe_user)

    except Exception as e:
        return {
            "ok": False,
            "error": str(e)
        }


@app.websocket("/ws/stream")
async def websocket_stream(ws: WebSocket):
    await ws.accept()

    session_id = str(uuid.uuid4())
    temp_path = os.path.join(SAVE_DIR, f"ws_{session_id}.webm")
    user_id = "user"
    file_handle = None

    try:
        file_handle = open(temp_path, "wb")
        await ws.send_json({"type": "ready"})

        while True:
            message = await ws.receive()

            if message.get("text") is not None:
                text = message["text"]

                if text.startswith("USER_ID:"):
                    user_id = text.split(":", 1)[1].strip() or "user"
                    continue

                if text == "END":
                    break

            if message.get("bytes") is not None:
                file_handle.write(message["bytes"])

        file_handle.close()
        file_handle = None

        result = await process_saved_video(temp_path, user_id=user_id)
        await ws.send_json({"type": "result", "data": result})

    except WebSocketDisconnect:
        print("WebSocket disconnected")

    except Exception as e:
        print("WebSocket error:", e)
        try:
            await ws.send_json({"type": "error", "error": str(e)})
        except Exception:
            pass

    finally:
        try:
            if file_handle:
                file_handle.close()
        except Exception:
            pass