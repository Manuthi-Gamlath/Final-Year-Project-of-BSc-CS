# app.py
# Run:
# uvicorn app:app --host 0.0.0.0 --port 8891

import os
import uuid
import json
import re
import time
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel

from media_utils import extract_muted_video, extract_audio,extract_uniform_sampled_video_opencv
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

# Whisper
whisper_model = WhisperModel("base", device="cpu", compute_type="int8")


# -----------------------------
# HELPERS
# -----------------------------
def now():
    return time.perf_counter()


def elapsed_ms(start_time: float) -> float:
    return round((time.perf_counter() - start_time) * 1000, 2)


def transcribe_audio(audio_path: str) -> str:
    segments, _ = whisper_model.transcribe(audio_path)
    return " ".join(s.text.strip() for s in segments if s.text.strip()).strip()


def classify_emotion(text):
    t = str(text).lower()

    if any(k in t for k in ["happy", "joy", "smile", "cheerful", "glad"]):
        return "happy"
    if any(k in t for k in ["angry", "mad", "furious", "annoy"]):
        return "angry"
    if any(k in t for k in ["sad", "cry", "unhappy", "down"]):
        return "sad"
    if any(k in t for k in ["fear", "scared", "afraid", "nervous", "worried"]):
        return "fear"
    if any(k in t for k in ["disgust", "gross", "repulse"]):
        return "disgust"
    if any(k in t for k in ["surprise", "surprised", "shocked", "amazed"]):
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
    if any(k in t for k in ["agree", "agreeing", "yes", "approval"]):
        return "Agreeing"
    if any(k in t for k in ["acknowledge", "acknowledging", "ok", "okay", "understand", "listen"]):
        return "Acknowledging"
    if any(k in t for k in ["clap", "clapping"]):
        return "Clap"

    return "None"


def default_message(raw_text: str = "", error_text: str = ""):
    return {
        "utterance": "",
        "age_range": "unknown",
        "gender": "unknown",
        "emotion": "neutral",
        "openness": 0.5,
        "conscientiousness": 0.5,
        "extraversion": 0.5,
        "agreeableness": 0.5,
        "neuroticism": 0.5,
        "ethnicity": "unknown",
        "agent_response": "Sorry, I could not understand properly.",
        "avatar_emotion": "neutral",
        "mixamo_gesture": "None",
        "raw": raw_text,
        "error": error_text,
    }


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
            return default_message(raw_text=text, error_text="empty model output")

        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return normalize_server_message(parsed)
        except Exception:
            pass

        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, dict):
                    return normalize_server_message(parsed)
            except Exception:
                pass

        return default_message(raw_text=text, error_text="invalid json format")

    except Exception as e:
        return default_message(raw_text=text, error_text=str(e))


def build_question(text: str):
    safe_text = str(text or "").replace("{", "(").replace("}", ")")

    return f"""
Analyze the video and the user utterance.

User utterance: "{safe_text}"

Return ONLY JSON.

Tasks:
- detect emotion of the person from: neutral, happy, sad, angry, fear, disgust, surprised
- detect gesture of the person from: Waving, Nod, Thinking, Agreeing, Acknowledging, Clap, None
- ethnicity is from: Sri lankan, chineese, black, unknown
- estimate lightweight personality scores from 0.00 to 1.00
- generate one short empathetic reply
- select matching avatar_emotion
- mixamo_gesture is the response to what person does

JSON schema:
{{
  "utterance": "{safe_text}",
  "age_range": "child|teen|young_adult|adult|older_adult|unknown",
  "gender": "male|female|unknown",
  "emotion": "neutral|happy|sad|angry|fear|disgust|surprised",
  "openness": 0.0,
  "conscientiousness": 0.0,
  "extraversion": 0.0,
  "agreeableness": 0.0,
  "neuroticism": 0.0,
  "ethnicity": "chineese|black|Sri lankan|unknown",
  "agent_response": "short empathetic reply",
  "avatar_emotion": "neutral|happy|sad|angry|fear|disgust|surprised",
  "mixamo_gesture": "Waving|Nod|Thinking|Agreeing|Acknowledging|Clap|None"
}}

Rules:
- JSON only
- no markdown
- no extra text
- emotion must reflect the visible user emotion
- agent_response must be socially aware
- avatar_emotion must align with the detected emotion
- mixamo_gesture must be one of the allowed values only
"""


async def process_saved_video(saved_path: str, user_id: str = "user"):
    muted_video_path = None
    audio_path = None
    timings = {}
    total_start = now()

    try:
        t = now()
        muted_video_path = extract_uniform_sampled_video_opencv(saved_path, DERIV_DIR,n_frames=16)
        timings["extract_muted_video_ms"] = elapsed_ms(t)

        t = now()
        audio_path = extract_audio(saved_path, DERIV_DIR, sr=16000)
        timings["extract_audio_ms"] = elapsed_ms(t)

        t = now()
        text = transcribe_audio(audio_path)
        timings["transcribe_audio_ms"] = elapsed_ms(t)

        if not text:
            text = ""

        print("Transcription:", text)

        t = now()
        question = build_question(text)
        timings["build_question_ms"] = elapsed_ms(t)

        t = now()
        output = infer_video(
            video_path=muted_video_path,
            num_segments=16,
            audio_path=audio_path,
            question=question
        )
        timings["infer_video_ms"] = elapsed_ms(t)

        print("Raw output:", output)

        t = now()
        parsed = parse_json_response(output)
        timings["parse_json_response_ms"] = elapsed_ms(t)

        timings["total_processing_ms"] = elapsed_ms(total_start)

        print("Timing summary:", timings)

        return {
            "ok": True,
            "message": parsed,
            "raw": output,
            "transcription": text,
            "saved_path": saved_path,
            "timings_ms": timings
        }

    except Exception as e:
        timings["total_processing_ms"] = elapsed_ms(total_start)
        print("process_saved_video error:", e)
        print("Timing summary:", timings)

        return {
            "ok": False,
            "error": str(e),
            "message": default_message(raw_text="", error_text=str(e)),
            "transcription": "",
            "saved_path": saved_path,
            "timings_ms": timings
        }


# -----------------------------
# STARTUP
# -----------------------------
@app.on_event("startup")
def startup():
    t = now()
    load_model_once(True)
    print(f"Model startup load time: {elapsed_ms(t)} ms")


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
    route_start = now()

    try:
        safe_user = (user_id or "user").replace("/", "_")
        safe_name = (file.filename or "upload.mp4").replace("/", "_")
        filename = f"{safe_user}_{safe_name}"
        saved_path = os.path.join(SAVE_DIR, filename)

        t = now()
        data = await file.read()
        read_file_ms = elapsed_ms(t)

        t = now()
        with open(saved_path, "wb") as f:
            f.write(data)
        save_file_ms = elapsed_ms(t)

        result = await process_saved_video(saved_path, user_id=safe_user)
        result["route_timings_ms"] = {
            "read_upload_file_ms": read_file_ms,
            "save_upload_file_ms": save_file_ms,
            "total_upload_route_ms": elapsed_ms(route_start)
        }
        return result

    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "message": default_message(raw_text="", error_text=str(e)),
            "route_timings_ms": {
                "total_upload_route_ms": elapsed_ms(route_start)
            }
        }


@app.websocket("/ws/stream")
async def websocket_stream(ws: WebSocket):
    await ws.accept()

    session_start = now()
    session_id = "dhanuja_session"  # str(uuid.uuid4())
    temp_path = os.path.join(SAVE_DIR, f"ws_{session_id}.webm")
    user_id = "user"
    file_handle = None
    receive_start = None

    try:
        file_handle = open(temp_path, "wb")
        await ws.send_json({"type": "ready"})
        receive_start = now()

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
                chunk = message["bytes"]
                if chunk:
                    file_handle.write(chunk)

        receive_stream_ms = elapsed_ms(receive_start)

        if file_handle:
            file_handle.close()
            file_handle = None

        process_start = now()
        result = await process_saved_video(temp_path, user_id=user_id)
        processing_after_stream_ms = elapsed_ms(process_start)

        result["ws_timings_ms"] = {
            "receive_stream_ms": receive_stream_ms,
            "processing_after_stream_ms": processing_after_stream_ms,
            "total_websocket_session_ms": elapsed_ms(session_start)
        }

        await ws.send_json({"type": "result", "data": result})

        try:
            await ws.close()
        except Exception:
            pass

    except WebSocketDisconnect:
        print("WebSocket disconnected")

    except Exception as e:
        print("WebSocket error:", e)
        try:
            await ws.send_json({
                "type": "error",
                "error": str(e),
                "ws_timings_ms": {
                    "total_websocket_session_ms": elapsed_ms(session_start)
                }
            })
        except Exception:
            pass
        try:
            await ws.close()
        except Exception:
            pass

    finally:
        try:
            if file_handle:
                file_handle.close()
        except Exception:
            pass