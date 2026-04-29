# app.py
# Run: uvicorn app:app --host 0.0.0.0 --port 8891

import os
import json
from pathlib import Path

import cv2
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
from deepface import DeepFace

from media_utils import extract_muted_video, extract_audio


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

# Whisper
whisper_model = WhisperModel("base", device="cpu", compute_type="int8")


# -----------------------------
# HELPERS
# -----------------------------
def transcribe_audio(audio_path: str) -> str:
    segments, _ = whisper_model.transcribe(audio_path)
    return " ".join(s.text.strip() for s in segments if s.text.strip()).strip()


def classify_emotion(text: str) -> str:
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
    if any(k in t for k in ["surprise", "surprised", "shocked", "amazed"]):
        return "surprised"

    return "neutral"


def classify_gesture(text: str) -> str:
    t = str(text).lower()

    if any(k in t for k in ["wave", "waving", "hello", "hi there", "greet"]):
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


def age_to_range(age_value) -> str:
    try:
        age = int(round(float(age_value)))
    except Exception:
        return "unknown"

    if age < 13:
        return "child"
    if age < 20:
        return "13-19"
    if age < 30:
        return "20-29"
    if age < 40:
        return "30-39"
    if age < 50:
        return "40-49"
    if age < 60:
        return "50-59"
    return "60+"


def extract_middle_frame(video_path: str, out_dir: str) -> str | None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        return None

    middle_index = frame_count // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_index)

    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        return None

    frame_path = os.path.join(out_dir, "deepface_frame.jpg")
    cv2.imwrite(frame_path, frame)
    return frame_path


def analyze_face_with_deepface(frame_path: str) -> dict:
    """
    DeepFace facial attribute analysis using:
    actions = ['age', 'gender', 'race', 'emotion']
    """
    try:
        result = DeepFace.analyze(
            img_path=frame_path,
            actions=["age", "gender", "race", "emotion"],
            enforce_detection=False,
            detector_backend="retinaface",
            silent=True,
        )

        # DeepFace may return a list or dict
        if isinstance(result, list):
            result = result[0] if result else {}

        age = result.get("age", "unknown")

        # gender
        gender = "unknown"
        if "dominant_gender" in result:
            gender = str(result["dominant_gender"]).lower()
        elif "gender" in result:
            g = result["gender"]
            if isinstance(g, dict) and g:
                gender = str(max(g, key=g.get)).lower()
            else:
                gender = str(g).lower()

        # emotion
        emotion = "neutral"
        if "dominant_emotion" in result:
            emotion = classify_emotion(result["dominant_emotion"])
        elif "emotion" in result:
            e = result["emotion"]
            if isinstance(e, dict) and e:
                emotion = classify_emotion(max(e, key=e.get))

        # race
        race = "unknown"
        if "dominant_race" in result:
            race = str(result["dominant_race"])
        elif "race" in result:
            r = result["race"]
            if isinstance(r, dict) and r:
                race = str(max(r, key=r.get))

        return {
            "age": age,
            "age_range": age_to_range(age),
            "gender": gender,
            "race": race,
            "emotion": emotion,
            "raw_deepface": result,
        }

    except Exception as e:
        print("DeepFace error:", str(e))
        return {
            "age": "unknown",
            "age_range": "unknown",
            "gender": "unknown",
            "race": "unknown",
            "emotion": "neutral",
            "raw_deepface": {},
            "deepface_error": str(e),
        }


def generate_agent_response_with_llm(user_text: str, face_info: dict) -> dict:
    """
    Replace this later with your real LLM call if needed.
    For now, this produces a clean conversational response
    using the detected DeepFace attributes.
    """

    emotion = classify_emotion(face_info.get("emotion", "neutral"))
    age_range = face_info.get("age_range", "unknown")
    gender = face_info.get("gender", "unknown")
    race = face_info.get("race", "unknown")

    # simple social cue
    t = (user_text or "").strip().lower()
    social_signals = "None"
    gesture = "None"

    if any(k in t for k in ["hello", "hi", "hey"]):
        social_signals = "greeting"
        gesture = "Waving"
    elif any(k in t for k in ["yes", "yeah", "correct", "right"]):
        social_signals = "agreement"
        gesture = "Agreeing"
    elif any(k in t for k in ["ok", "okay", "understand"]):
        social_signals = "acknowledgement"
        gesture = "Acknowledging"

    # response based on detected emotion
    if emotion == "happy":
        agent_response = "You look happy. It is nice to talk with you."
        avatar_emotion = "happy"
    elif emotion == "sad":
        agent_response = "You seem a little sad. I am here to talk with you."
        avatar_emotion = "sad"
    elif emotion == "angry":
        agent_response = "You seem upset. I will respond calmly and clearly."
        avatar_emotion = "neutral"
    elif emotion == "fear":
        agent_response = "You seem worried. I am here to help."
        avatar_emotion = "sad"
    elif emotion == "disgust":
        agent_response = "I understand your reaction. Tell me what happened."
        avatar_emotion = "neutral"
    elif emotion == "surprised":
        agent_response = "You look surprised. What would you like to know?"
        avatar_emotion = "surprised"
    else:
        agent_response = "Hello. How can I help you today?"
        avatar_emotion = "neutral"

    if user_text and user_text != "[no speech detected]":
        agent_response = f"{agent_response} You said: {user_text}"

    return {
        "utterance": user_text if user_text else "",
        "age_range": age_range,
        "gender": gender,
        "race": race,
        "emotion": emotion,
        "openness": 0.5,
        "conscientiousness": 0.5,
        "extraversion": 0.5,
        "agreeableness": 0.5,
        "neuroticism": 0.5,
        "language_or_cultural_cue": race,
        "social_signals": social_signals,
        "agent_response": agent_response,
        "avatar_emotion": classify_emotion(avatar_emotion),
        "mixamo_gesture": classify_gesture(gesture),
    }


# -----------------------------
# STARTUP
# -----------------------------
@app.on_event("startup")
def startup():
    print("Server started.")
    print("Whisper loaded.")
    print("DeepFace will load on first use.")


# -----------------------------
# ROUTES
# -----------------------------
@app.get("/health")
def health():
    return {
        "ok": True,
        "whisper_loaded": True,
        "deepface_ready": True,
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
        # TRANSCRIBE AUDIO
        # -----------------------------
        text = transcribe_audio(audio_path)
        if not text:
            text = "[no speech detected]"
        print("Transcription:", text)

        # -----------------------------
        # DEEPFACE ANALYSIS
        # -----------------------------
        frame_path = extract_middle_frame(muted_video_path or saved_path, DERIV_DIR)

        if frame_path is None:
            face_info = {
                "age": "unknown",
                "age_range": "unknown",
                "gender": "unknown",
                "race": "unknown",
                "emotion": "neutral",
                "raw_deepface": {},
                "deepface_error": "Could not extract frame from video",
            }
        else:
            face_info = analyze_face_with_deepface(frame_path)

        print("DeepFace output:", json.dumps(face_info, indent=2))

        # -----------------------------
        # LLM / RESPONSE GENERATION
        # -----------------------------
        parsed = generate_agent_response_with_llm(text, face_info)

        # keep DeepFace values aligned
        parsed["age_range"] = face_info.get("age_range", "unknown")
        parsed["gender"] = face_info.get("gender", "unknown")
        parsed["race"] = face_info.get("race", "unknown")
        parsed["emotion"] = classify_emotion(face_info.get("emotion", "neutral"))
        parsed["avatar_emotion"] = classify_emotion(parsed.get("avatar_emotion", "neutral"))
        parsed["mixamo_gesture"] = classify_gesture(parsed.get("mixamo_gesture", "None"))

        # -----------------------------
        # RESPONSE
        # -----------------------------
        return {
            "ok": True,
            "message": parsed,
            "transcription": text,
            "deepface": face_info,
            "saved_path": saved_path,
            "frame_path": frame_path,
        }

    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
        }