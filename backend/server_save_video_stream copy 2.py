# app.py
# Run:
# uvicorn app:app --host 0.0.0.0 --port 8891

import os
import json
import re
import time
import asyncio
from pathlib import Path
from threading import Thread

import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, TextIteratorStreamer

from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel

from media_utils import extract_audio, extract_uniform_sampled_video_opencv
from modeling_internvl_chat_hico2 import InternVLChatModel
from conversation import get_conv_template


# -----------------------------
# CONFIG
# -----------------------------
SAVE_DIR = "uploads"
DERIV_DIR = os.path.join(SAVE_DIR, "derived")
MODEL_PATH = "OpenGVLab/InternVideo2_5_Chat_8B"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

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

whisper_model = WhisperModel("base", device="cpu", compute_type="int8")

tokenizer = None
model = None

generation_config = {
    "do_sample": False,
    "temperature": 0.0,
    "max_new_tokens": 256,
    "top_p": 0.1,
    "num_beams": 1,
}


# -----------------------------
# HELPERS
# -----------------------------
def now():
    return time.perf_counter()


def elapsed_ms(start_time: float) -> float:
    return round((time.perf_counter() - start_time) * 1000, 2)


def is_model_loaded() -> bool:
    return model is not None and tokenizer is not None


def load_model_once(force: bool = False):
    global tokenizer, model

    if is_model_loaded() and not force:
        return

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = InternVLChatModel.from_pretrained(MODEL_PATH, trust_remote_code=True).cuda()
    model.eval()


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


def parse_meta_json_response(text: str):
    try:
        raw = str(text).strip()
        if not raw:
            return {
                "avatar_emotion": "neutral",
                "mixamo_gesture": "None",
            }

        parsed = None
        try:
            parsed = json.loads(raw)
        except Exception:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                parsed = json.loads(match.group(0))

        if not isinstance(parsed, dict):
            raise ValueError("invalid metadata json")

        avatar_emotion = classify_emotion(parsed.get("avatar_emotion", "neutral"))
        mixamo_gesture = classify_gesture(parsed.get("mixamo_gesture", "None"))

        return {
            "avatar_emotion": avatar_emotion,
            "mixamo_gesture": mixamo_gesture,
        }
    except Exception:
        return {
            "avatar_emotion": "neutral",
            "mixamo_gesture": "None",
        }


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
""".strip()


def build_meta_question(text: str):
    safe_text = str(text or "").replace("{", "(").replace("}", ")").strip()
    return f"""
Analyze the video and the user utterance.

User utterance: "{safe_text}"

Return ONLY valid JSON with exactly these keys:
{{
  "avatar_emotion": "neutral|happy|sad|angry|fear|disgust|surprised",
  "mixamo_gesture": "Waving|Nod|Thinking|Agreeing|Acknowledging|Clap|None"
}}

Rules:
- JSON only
- no markdown
- no extra text
- avatar_emotion must match the visible emotion
- mixamo_gesture must match the visible social response
""".strip()


def build_reply_question(text: str, avatar_emotion: str, mixamo_gesture: str):
    safe_text = str(text or "").replace("{", "(").replace("}", ")").strip()
    safe_emotion = str(avatar_emotion or "neutral")
    safe_gesture = str(mixamo_gesture or "None")

    return f"""
You are a warm voice assistant.

User utterance: "{safe_text}"
Detected avatar emotion: "{safe_emotion}"
Detected gesture: "{safe_gesture}"

Write only the assistant reply as plain text.

Rules:
- plain text only
- no JSON
- no markdown
- short, natural, empathetic
- 1 to 3 sentences
""".strip()


# -----------------------------
# PREPROCESSING HELPERS
# -----------------------------
def build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height

    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)

        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio

    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []

    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)

    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)

    return processed_images


def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000

    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)

    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices


def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=16, get_frame_by_duration=False):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list = []
    num_patches_list = []
    transform = build_transform(input_size=input_size)

    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)

    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert("RGB")
        img_tiles = dynamic_preprocess(
            img, image_size=input_size, use_thumbnail=True, max_num=max_num
        )
        pixel_values = [transform(tile) for tile in img_tiles]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)

    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list


# -----------------------------
# MODEL GENERATION
# -----------------------------
def generate_stream_sync(video_path: str, question: str, num_segments: int = 16):
    if not is_model_loaded():
        load_model_once()

    with torch.no_grad():
        pixel_values, num_patches_list = load_video(
            video_path,
            num_segments=num_segments,
            max_num=1,
            get_frame_by_duration=False
        )
        pixel_values = pixel_values.to(model.device)

        video_prefix = "".join([f"Frame{i+1}: <image>\n" for i in range(len(num_patches_list))])
        full_question = question + "\n" + video_prefix

        template = get_conv_template(model.template)
        template.system_message = model.system_message
        template.append_message(template.roles[0], full_question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        img_start_token = "<img>"
        img_end_token = "</img>"
        img_context_token = "<IMG_CONTEXT>"

        model.img_context_token_id = tokenizer.convert_tokens_to_ids(img_context_token)

        for num_patches in num_patches_list:
            image_tokens = (
                img_start_token
                + img_context_token * model.num_image_token * num_patches
                + img_end_token
            )
            query = query.replace("<image>", image_tokens, 1)

        model_inputs = tokenizer(query, return_tensors="pt")
        input_ids = model_inputs["input_ids"].to(model.device)
        attention_mask = model_inputs["attention_mask"].to(model.device)

        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())

        streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )

        generation_kwargs = dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            streamer=streamer,
            eos_token_id=eos_token_id,
            **generation_config
        )

        thread = Thread(target=model.generate, kwargs=generation_kwargs, daemon=True)
        thread.start()

        for new_text in streamer:
            yield new_text

        thread.join()


def generate_text_sync(video_path: str, question: str, num_segments: int = 16):
    text = ""
    for chunk in generate_stream_sync(video_path, question, num_segments=num_segments):
        text += chunk
    return text.strip()


def next_chunk_or_none(iterator):
    try:
        return next(iterator)
    except StopIteration:
        return None


# -----------------------------
# PROCESSORS
# -----------------------------
async def process_saved_video(saved_path: str, user_id: str = "user"):
    muted_video_path = None
    audio_path = None
    timings = {}
    total_start = now()

    try:
        t = now()
        muted_video_path = extract_uniform_sampled_video_opencv(saved_path, DERIV_DIR, n_frames=16)
        timings["extract_muted_video_ms"] = elapsed_ms(t)

        t = now()
        audio_path = extract_audio(saved_path, DERIV_DIR, sr=16000)
        timings["extract_audio_ms"] = elapsed_ms(t)

        t = now()
        text = transcribe_audio(audio_path)
        timings["transcribe_audio_ms"] = elapsed_ms(t)

        if not text:
            text = ""

        t = now()
        question = build_question(text)
        timings["build_question_ms"] = elapsed_ms(t)

        t = now()
        output = await asyncio.to_thread(
            generate_text_sync,
            muted_video_path,
            question,
            16
        )
        timings["infer_video_ms"] = elapsed_ms(t)

        t = now()
        parsed = parse_json_response(output)
        timings["parse_json_response_ms"] = elapsed_ms(t)

        timings["total_processing_ms"] = elapsed_ms(total_start)

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
        return {
            "ok": False,
            "error": str(e),
            "message": default_message(raw_text="", error_text=str(e)),
            "transcription": "",
            "saved_path": saved_path,
            "timings_ms": timings
        }


async def process_saved_video_with_live_events(saved_path: str, ws: WebSocket, user_id: str = "user"):
    muted_video_path = None
    audio_path = None
    timings = {}
    total_start = now()

    try:
        t = now()
        muted_video_path = extract_uniform_sampled_video_opencv(saved_path, DERIV_DIR, n_frames=16)
        timings["extract_muted_video_ms"] = elapsed_ms(t)
        await ws.send_json({"type": "status", "stage": "video_prepared"})

        t = now()
        audio_path = extract_audio(saved_path, DERIV_DIR, sr=16000)
        timings["extract_audio_ms"] = elapsed_ms(t)
        await ws.send_json({"type": "status", "stage": "audio_extracted"})

        t = now()
        text = transcribe_audio(audio_path)
        timings["transcribe_audio_ms"] = elapsed_ms(t)
        text = text or ""
        await ws.send_json({"type": "transcription", "text": text})

        t = now()
        meta_question = build_meta_question(text)
        meta_raw = await asyncio.to_thread(
            generate_text_sync,
            muted_video_path,
            meta_question,
            16
        )
        meta = parse_meta_json_response(meta_raw)
        timings["meta_generation_ms"] = elapsed_ms(t)

        await ws.send_json({
            "type": "meta",
            "data": meta
        })

        t = now()
        reply_question = build_reply_question(
            text=text,
            avatar_emotion=meta["avatar_emotion"],
            mixamo_gesture=meta["mixamo_gesture"],
        )

        reply_accum = ""
        stream_iter = generate_stream_sync(muted_video_path, reply_question, 16)

        while True:
            chunk = await asyncio.to_thread(next_chunk_or_none, stream_iter)

            if chunk is None:
                break

            reply_accum += chunk
            await ws.send_json({
                "type": "partial_text",
                "text": chunk,
                "full_text": reply_accum
            })
            await asyncio.sleep(0)

        timings["reply_generation_ms"] = elapsed_ms(t)
        timings["total_processing_ms"] = elapsed_ms(total_start)

        final_message = {
            "agent_response": reply_accum.strip() or "Sorry, I could not understand properly.",
            "avatar_emotion": meta["avatar_emotion"],
            "mixamo_gesture": meta["mixamo_gesture"],
        }

        return {
            "ok": True,
            "message": final_message,
            "transcription": text,
            "saved_path": saved_path,
            "timings_ms": timings,
        }

    except Exception as e:
        timings["total_processing_ms"] = elapsed_ms(total_start)
        return {
            "ok": False,
            "error": str(e),
            "message": {
                "agent_response": "Sorry, I could not understand properly.",
                "avatar_emotion": "neutral",
                "mixamo_gesture": "None",
            },
            "transcription": "",
            "saved_path": saved_path,
            "timings_ms": timings,
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
    session_id = "dhanuja_session"
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
        result = await process_saved_video_with_live_events(temp_path, ws, user_id=user_id)
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