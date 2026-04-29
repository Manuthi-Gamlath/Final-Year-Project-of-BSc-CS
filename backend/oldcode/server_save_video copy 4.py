# app.py
# Run with:
# uvicorn server_save_video:app --host 0.0.0.0 --port 8891

import json
import os
import re
from pathlib import Path
from typing import Any, Dict

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
    allow_origins=["*"],   # tighten this later in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Whisper model loads once
whisper_model = WhisperModel("base", device="cpu", compute_type="int8")


# -----------------------------
# HELPERS
# -----------------------------
def transcribe_audio(audio_path: str) -> str:
    segments, info = whisper_model.transcribe(audio_path)
    full_text = " ".join(segment.text.strip() for segment in segments if segment.text.strip())
    return full_text.strip()


def sanitize_filename(name: str) -> str:
    return (name or "").replace("/", "_").replace("\\", "_").strip()


def normalize_float(value: Any, default: float = 0.5) -> float:
    try:
        num = float(value)
        if num < 0:
            return 0.0
        if num > 1:
            return 1.0
        return num
    except Exception:
        return default


def normalize_text(value: Any, default: str = "None") -> str:
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    return text


def extract_json_object(text: str) -> Dict[str, Any]:
    """
    Try to extract a JSON object even if the model wrapped it with extra text.
    """
    if not text:
        raise ValueError("Empty model output")

    text = text.strip()

    # First try direct JSON parse
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Try to find the first {...} block
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        candidate = match.group(0)
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            return obj

    raise ValueError(f"Could not parse JSON from model output: {text}")


def make_fallback_response(user_utterance: str, raw_output: str = "") -> Dict[str, Any]:
    """
    Safe fallback if model output is malformed.
    """
    utterance = normalize_text(user_utterance, default="")
    agent_response = "Hello, how can I help you?"

    if utterance:
        low = utterance.lower()
        if "how are you" in low:
            agent_response = "I'm good, thanks for asking."
        elif "hello" in low or "hi" in low:
            agent_response = "Hello, how are you?"
        else:
            agent_response = "I heard you. How can I help you?"

    return {
        "utterance": utterance,
        "age_range": "20-30",
        "gender": "unknown",
        "emotion": "neutral",
        "openness": 0.5,
        "conscientiousness": 0.5,
        "extraversion": 0.5,
        "agreeableness": 0.5,
        "neuroticism": 0.5,
        "language_or_cultural_cue": "None",
        "social_signals": "None",
        "agent_response": agent_response,
        "avatar_emotion": "neutral",
        "mixamo_gesture": "None",
        "raw_output": raw_output,
        "parse_error": True,
    }


def normalize_model_response(parsed: Dict[str, Any], user_utterance: str, raw_output: str) -> Dict[str, Any]:
    """
    Force the response into a stable schema for the frontend.
    """
    return {
        "utterance": normalize_text(parsed.get("utterance"), default=user_utterance),
        "age_range": normalize_text(parsed.get("age_range"), default="unknown"),
        "gender": normalize_text(parsed.get("gender"), default="unknown"),
        "emotion": normalize_text(parsed.get("emotion"), default="neutral"),
        "openness": normalize_float(parsed.get("openness"), default=0.5),
        "conscientiousness": normalize_float(parsed.get("conscientiousness"), default=0.5),
        "extraversion": normalize_float(parsed.get("extraversion"), default=0.5),
        "agreeableness": normalize_float(parsed.get("agreeableness"), default=0.5),
        "neuroticism": normalize_float(parsed.get("neuroticism"), default=0.5),
        "language_or_cultural_cue": normalize_text(parsed.get("language_or_cultural_cue"), default="None"),
        "social_signals": normalize_text(parsed.get("social_signals"), default="None"),
        "agent_response": normalize_text(parsed.get("agent_response"), default="Hello, how can I help you?"),
        "avatar_emotion": normalize_text(parsed.get("avatar_emotion"), default="neutral"),
        "mixamo_gesture": normalize_text(parsed.get("mixamo_gesture"), default="None"),
        "raw_output": raw_output,
        "parse_error": False,
    }


def build_question(user_utterance: str) -> str:
    return f"""
You are a multimodal AI assistant for the SELA system.

Analyze the given video together with this user utterance:
{json.dumps(user_utterance)}

Return ONLY valid JSON.
Do not return markdown.
Do not return explanations.
Do not return CSV.
Do not return key-value lines.

Return exactly these keys:
utterance
age_range
gender
emotion
openness
conscientiousness
extraversion
agreeableness
neuroticism
language_or_cultural_cue
social_signals
agent_response
avatar_emotion
mixamo_gesture

Rules:
- utterance, age_range, gender, emotion, language_or_cultural_cue, social_signals,
  agent_response, avatar_emotion, mixamo_gesture must be strings.
- openness, conscientiousness, extraversion, agreeableness, neuroticism must be numbers between 0 and 1.
- agent_response must be one short natural sentence.
- avatar_emotion should be appropriate for a facial avatar.
- mixamo_gesture should be a short gesture label such as "Idle", "Talking", "Nod", "Waving", or "None".
- social_signals should be a short string such as "greeting", "question", "hesitation", "engaged", or "None".

Example output:
{{
  "utterance": "How are you?",
  "age_range": "20-30",
  "gender": "male",
  "emotion": "neutral",
  "openness": 0.5,
  "conscientiousness": 0.7,
  "extraversion": 0.3,
  "agreeableness": 0.6,
  "neuroticism": 0.2,
  "language_or_cultural_cue": "None",
  "social_signals": "greeting",
  "agent_response": "I'm good, thanks for asking.",
  "avatar_emotion": "neutral",
  "mixamo_gesture": "Waving"
}}
""".strip()


# -----------------------------
# STARTUP: LOAD MODEL ONCE
# -----------------------------
@app.on_event("startup")
def _startup():
    load_model_once()


# -----------------------------
# ROUTES
# -----------------------------
@app.get("/health")
def health():
    return {
        "ok": True,
        "message": "backend is reachable",
        "model_loaded": is_model_loaded(),
    }


@app.post("/upload")
async def upload_video(
    file: UploadFile = File(...),
    seconds: str = Form(None),
    user_id: str = Form(None),
):
    try:
        safe_user = sanitize_filename(user_id or "user")
        safe_name = sanitize_filename(file.filename or "upload.webm")
        filename = f"{safe_user}_{safe_name}"
        saved_path = os.path.join(SAVE_DIR, filename)

        # Save uploaded file
        data = await file.read()
        with open(saved_path, "wb") as f:
            f.write(data)

        size_kb = len(data) / 1024.0

        # Derive media
        muted_video_path = extract_muted_video(saved_path, DERIV_DIR)
        audio_path = extract_audio(saved_path, DERIV_DIR, sr=16000)

        # IMPORTANT FIX:
        # use the real extracted audio path instead of a hardcoded path
        text = transcribe_audio(audio_path)
        print("Transcription:", text)

        user_utterance = text or ""
        question = build_question(user_utterance)

        # Model inference
        raw_output = infer_video(
            video_path=muted_video_path,
            audio_path=audio_path,
            question=question,
        )

        print("Raw model output:", raw_output)

        try:
            print("---------------VLM replied----------------")
            print(raw_output)
            parsed = extract_json_object(raw_output)
            message = normalize_model_response(parsed, user_utterance=user_utterance, raw_output=raw_output)
        except Exception as parse_err:
            print("JSON parse failed:", parse_err)
            message = make_fallback_response(user_utterance=user_utterance, raw_output=raw_output)

        return {
            "ok": True,
            "message": message,
            "saved_path": saved_path,
            "muted_video_path": muted_video_path,
            "audio_path": audio_path,
            "transcription": user_utterance,
            "seconds": seconds,
            "size_kb": round(size_kb, 2),
        }

    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
        }