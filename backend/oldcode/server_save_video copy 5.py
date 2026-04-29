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
# SUPPORTED LABELS
# -----------------------------
ALLOWED_EMOTIONS = {
    "neutral",
    "happy",
    "angry",
    "sad",
    "fear",
    "disgust",
    "surprised",
}

ALLOWED_GESTURES = {
    "None",
    "Waving",
    "Nod",
    "Thinking",
    "Agreeing",
    "Acknowledging",
}

ALLOWED_SOCIAL_SIGNALS = {
    "greeting",
    "question",
    "hesitation",
    "engaged",
    "unclear",
    "None",
}


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


def normalize_choice(value: Any, allowed: set[str], default: str) -> str:
    text = normalize_text(value, default=default)

    for item in allowed:
        if text.lower() == item.lower():
            return item

    aliases = {
        # emotion aliases
        "calm": "neutral",
        "neutrality": "neutral",
        "joy": "happy",
        "excited": "happy",
        "mad": "angry",
        "afraid": "fear",
        "scared": "fear",
        "shock": "surprised",

        # gesture aliases
        "idle": "None",
        "no gesture": "None",
        "no clear gesture": "None",
        "wave": "Waving",
        "waving hand": "Waving",
        "nodding": "Nod",
        "head nod": "Nod",
        "acknowledge": "Acknowledging",
        "talking": "Acknowledging",
        "agree": "Agreeing",
        "yes": "Agreeing",
        "thinking pose": "Thinking",
        "reflective": "Thinking",
    }

    mapped = aliases.get(text.lower())
    if mapped and mapped in allowed:
        return mapped

    return default


def extract_json_object(text: str) -> Dict[str, Any]:
    """
    Try to extract a JSON object even if the model wrapped it with extra text.
    """
    if not text:
        raise ValueError("Empty model output")

    text = text.strip()

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        candidate = match.group(0)
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            return obj

    raise ValueError(f"Could not parse JSON from model output: {text}")


def prepare_user_utterance(text: str) -> str:
    cleaned = (text or "").strip()
    if not cleaned:
        return "[no clear speech detected]"
    return cleaned


def infer_social_signal_from_text(user_utterance: str) -> str:
    u = (user_utterance or "").strip().lower()

    if not u or u == "[no clear speech detected]":
        return "unclear"

    if any(x in u for x in ["hello", "hi", "hey", "good morning", "good evening"]):
        return "greeting"

    if "?" in u or any(x in u for x in ["what", "why", "how", "can you", "could you", "do you"]):
        return "question"

    if any(x in u for x in ["hmm", "uh", "um", "not sure", "maybe", "i think"]):
        return "hesitation"

    return "engaged"


def is_generic_response(text: str) -> bool:
    t = (text or "").strip().lower()
    generic = {
        "",
        "hello, how can i help you?",
        "hello, how are you?",
        "i'm good, thanks for asking.",
        "i heard you. how can i help you?",
        "how can i assist you today?",
    }
    return t in generic


def choose_agent_response(user_utterance: str, social_signal: str, model_response: str) -> str:
    utterance = (user_utterance or "").strip()
    low = utterance.lower()

    if utterance == "[no clear speech detected]":
        return "I could not hear you clearly. Could you say that again?"

    if not is_generic_response(model_response):
        return model_response.strip()

    if social_signal == "greeting":
        if "how are you" in low:
            return "I'm doing well, thanks for asking."
        return "Hello. How can I help you today?"

    if social_signal == "question":
        if "how are you" in low:
            return "I'm doing well, thanks for asking."
        return "I heard your question. Let me help with that."

    if social_signal == "hesitation":
        return "Take your time. I'm here to help."

    return "I heard you. How can I help you?"


def choose_avatar_emotion(
    user_utterance: str,
    detected_emotion: str,
    model_avatar_emotion: str,
    social_signal: str,
) -> str:
    """
    Keep avatar emotion compatible with visible user emotion.
    Prefer valid model choice first, otherwise mirror detected video emotion.
    """
    model_choice = normalize_choice(model_avatar_emotion, ALLOWED_EMOTIONS, default="neutral")
    detected = normalize_choice(detected_emotion, ALLOWED_EMOTIONS, default="neutral")

    if model_choice in ALLOWED_EMOTIONS and model_choice != "neutral":
        return model_choice

    if detected in ALLOWED_EMOTIONS:
        return detected

    return "neutral"


def choose_gesture(social_signal: str, model_gesture: str) -> str:
    """
    Prefer gesture inferred from video/model.
    If unclear, fallback to a simple social-signal-based gesture.
    """
    model_choice = normalize_choice(model_gesture, ALLOWED_GESTURES, default="None")
    if model_choice in ALLOWED_GESTURES and model_choice != "None":
        return model_choice

    if social_signal == "greeting":
        return "Waving"
    if social_signal == "question":
        return "Acknowledging"
    if social_signal == "hesitation":
        return "Thinking"

    return "None"


def make_fallback_response(user_utterance: str, raw_output: str = "") -> Dict[str, Any]:
    """
    Safe fallback if model output is malformed.
    """
    utterance = prepare_user_utterance(user_utterance)
    social_signal = infer_social_signal_from_text(utterance)

    detected_emotion = "neutral"
    agent_response = choose_agent_response(
        user_utterance=utterance,
        social_signal=social_signal,
        model_response="",
    )
    avatar_emotion = choose_avatar_emotion(
        user_utterance=utterance,
        detected_emotion=detected_emotion,
        model_avatar_emotion="neutral",
        social_signal=social_signal,
    )
    gesture = choose_gesture(
        social_signal=social_signal,
        model_gesture="None",
    )

    return {
        "utterance": utterance,
        "age_range": "20-30",
        "gender": "unknown",
        "emotion": detected_emotion,
        "openness": 0.5,
        "conscientiousness": 0.5,
        "extraversion": 0.5,
        "agreeableness": 0.5,
        "neuroticism": 0.5,
        "language_or_cultural_cue": "None",
        "social_signals": social_signal,
        "agent_response": agent_response,
        "avatar_emotion": avatar_emotion,
        "mixamo_gesture": gesture,
        "raw_output": raw_output,
        "parse_error": True,
    }


def normalize_model_response(parsed: Dict[str, Any], user_utterance: str, raw_output: str) -> Dict[str, Any]:
    """
    Force the response into a stable schema for the frontend.
    emotion should match the user's visible video emotion as much as possible.
    avatar_emotion should stay compatible with that emotion.
    mixamo_gesture should react to visible gesture/body language in the video.
    """
    utterance = normalize_text(parsed.get("utterance"), default=user_utterance)
    utterance = prepare_user_utterance(utterance)

    detected_emotion = normalize_choice(
        parsed.get("emotion"),
        ALLOWED_EMOTIONS,
        default="neutral",
    )

    social_signal = normalize_choice(
        parsed.get("social_signals"),
        ALLOWED_SOCIAL_SIGNALS,
        default=infer_social_signal_from_text(utterance),
    )

    agent_response = choose_agent_response(
        user_utterance=utterance,
        social_signal=social_signal,
        model_response=normalize_text(parsed.get("agent_response"), default=""),
    )

    avatar_emotion = choose_avatar_emotion(
        user_utterance=utterance,
        detected_emotion=detected_emotion,
        model_avatar_emotion=normalize_text(parsed.get("avatar_emotion"), default="neutral"),
        social_signal=social_signal,
    )

    mixamo_gesture = choose_gesture(
        social_signal=social_signal,
        model_gesture=normalize_text(parsed.get("mixamo_gesture"), default="None"),
    )

    return {
        "utterance": utterance,
        "age_range": normalize_text(parsed.get("age_range"), default="unknown"),
        "gender": normalize_text(parsed.get("gender"), default="unknown"),
        "emotion": detected_emotion,
        "openness": normalize_float(parsed.get("openness"), default=0.5),
        "conscientiousness": normalize_float(parsed.get("conscientiousness"), default=0.5),
        "extraversion": normalize_float(parsed.get("extraversion"), default=0.5),
        "agreeableness": normalize_float(parsed.get("agreeableness"), default=0.5),
        "neuroticism": normalize_float(parsed.get("neuroticism"), default=0.5),
        "language_or_cultural_cue": normalize_text(parsed.get("language_or_cultural_cue"), default="None"),
        "social_signals": social_signal,
        "agent_response": agent_response,
        "avatar_emotion": avatar_emotion,
        "mixamo_gesture": mixamo_gesture,
        "raw_output": raw_output,
        "parse_error": False,
    }


def build_question(user_utterance: str) -> str:
    safe_utterance = prepare_user_utterance(user_utterance)

    return f"""
You are a multimodal AI assistant for the SELA system.

Analyze the uploaded user video together with the transcribed utterance below.

Transcribed utterance:
{json.dumps(safe_utterance)}

Return ONLY valid JSON.
Do not return markdown.
Do not return explanations.
Do not return CSV.
Do not return key-value lines.
Do not include any text before or after the JSON object.

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
- utterance, age_range, gender, emotion, language_or_cultural_cue, social_signals, agent_response, avatar_emotion, mixamo_gesture must be strings.
- openness, conscientiousness, extraversion, agreeableness, neuroticism must be numbers between 0 and 1.
- Base the output on actual visible cues in the video and the utterance.
- emotion must reflect the user's visible facial emotion and body affect in the video as much as possible.
- avatar_emotion should be compatible with the user's visible emotion in the video.
- If uncertain, prefer mirroring the user's emotion rather than inventing a different one.
- mixamo_gesture must reflect the user's visible gesture or body language in the video.
- Only output a gesture if there is visible evidence in the video.
- If no clear gesture is visible, return "None".
- agent_response must be one short natural sentence that fits the user's utterance and emotional state.
- If the utterance is unclear, use visible social cues from the video if possible.
- Do not always output the same reply.

Allowed values:
- emotion must be one of:
  "neutral", "happy", "angry", "sad", "fear", "disgust", "surprised"
- avatar_emotion must be one of:
  "neutral", "happy", "angry", "sad", "fear", "disgust", "surprised"
- mixamo_gesture must be one of:
  "None", "Waving", "Nod", "Thinking", "Agreeing", "Acknowledging"
- social_signals should be one of:
  "greeting", "question", "hesitation", "engaged", "unclear", "None"

Gesture mapping guidance:
- wave / raising hand -> "Waving"
- head nod / yes-like motion -> "Nod"
- reflective pause / thoughtful pose -> "Thinking"
- agreement-like body language -> "Agreeing"
- mild conversational acknowledgment -> "Acknowledging"
- no clear visible gesture -> "None"

Additional instructions:
- Prefer video evidence over generic defaults.
- Do not use unsupported emotion labels.
- If there is no clear speech, say that politely in agent_response.

Return only one JSON object.
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

        # Transcribe extracted audio
        text = transcribe_audio(audio_path)
        print("Transcription:", text)

        user_utterance = prepare_user_utterance(text)
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
            message = normalize_model_response(
                parsed,
                user_utterance=user_utterance,
                raw_output=raw_output,
            )
        except Exception as parse_err:
            print("JSON parse failed:", parse_err)
            message = make_fallback_response(
                user_utterance=user_utterance,
                raw_output=raw_output,
            )

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