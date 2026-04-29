# model_service.py
import threading
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer

# ✅ Your model class (same import as your demo)
from modeling_internvl_chat_hico2 import InternVLChatModel

# -----------------------------
# CONFIG (same as your demo, but centralized)
# -----------------------------
MODEL_PATH = "OpenGVLab/InternVideo2_5_Chat_8B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEFAULT_GENERATION_CONFIG = dict(
    do_sample=False,
    temperature=0.0,
    max_new_tokens=1024,
    top_p=0.1,
    num_beams=1
)

# -----------------------------
# SINGLETON STATE
# -----------------------------
_model: Optional[InternVLChatModel] = None
_tokenizer: Optional[Any] = None
_lock = threading.Lock()  # prevents concurrent GPU inference collisions


# -----------------------------
# PREPROCESS (copied from your demo with tiny safety tweaks)
# -----------------------------
def build_transform(input_size: int):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD),
    ])
    return transform


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

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
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

    assert len(processed_images) == blocks

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
    seg_size = float(end_idx - start_idx) / float(num_segments)
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices


def get_num_frames_by_duration(duration: float) -> int:
    # copied logic from your demo
    local_num_frames = 4
    num_segments = int(duration // local_num_frames)
    if num_segments == 0:
        num_frames = local_num_frames
    else:
        num_frames = local_num_frames * num_segments

    num_frames = min(512, num_frames)
    num_frames = max(128, num_frames)
    return num_frames


def load_video(
    video_path: str,
    bound=None,
    input_size: int = 448,
    max_num: int = 1,
    num_segments: int = 60,
    get_frame_by_duration: bool = False
) -> Tuple[torch.Tensor, List[int]]:
    """
    Same behavior as your demo:
    returns pixel_values (cat over frames) and num_patches_list (patch tiles per frame).
    """
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)

    if get_frame_by_duration:
        duration = max_frame / fps
        num_segments = get_num_frames_by_duration(duration)

    # Safety: ensure num_segments > 0
    num_segments = int(max(1, num_segments))

    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)

    for frame_index in frame_indices:
        # clamp just in case
        frame_index = int(max(0, min(frame_index, max_frame)))
        img = Image.fromarray(vr[frame_index].asnumpy()).convert("RGB")
        tiles = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in tiles]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)

    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list


# -----------------------------
# MODEL LOADING (LOAD ONCE)
# -----------------------------
def load_model_once(use_half: bool = False) -> None:
    """
    ✅ Load model + tokenizer ONE TIME.
    - call this in FastAPI startup
    - reuse the same model for every request
    """
    global _model, _tokenizer
    if _model is not None and _tokenizer is not None:
        return

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    _model = InternVLChatModel.from_pretrained(MODEL_PATH, trust_remote_code=True)

    if use_half and DEVICE == "cuda":
        _model = _model.half()

    _model = _model.to(DEVICE)
    _model.eval()


def is_model_loaded() -> bool:
    return _model is not None and _tokenizer is not None


# -----------------------------
# INFERENCE (REUSE MODEL MANY TIMES)
# -----------------------------
@torch.no_grad()
def infer_video(
    video_path: str,
    audio_path: Optional[str] = None,  # not used by InternVideo2.5 in your demo; kept for API compatibility
    question: str = "detect objects in the video?",
    num_segments: int = 16,
    max_num_tiles_per_frame: int = 1,
    get_frame_by_duration: bool = False,
    input_size: int = 448,
    generation_config: Optional[Dict[str, Any]] = None,
) -> str:
    """
    ✅ Uses already-loaded model/tokenizer.
    ✅ Runs your exact pipeline:
       pixel_values, num_patches_list = load_video(...)
       video_prefix = "Frame1: <image> ..."
       output, history = model.chat(...)
    """
    if not is_model_loaded():
        raise RuntimeError("Model not loaded. Call load_model_once() at startup.")

    gen_cfg = generation_config or DEFAULT_GENERATION_CONFIG

    # If your pipeline isn't thread-safe, keep this lock
    with _lock:
        pixel_values, num_patches_list = load_video(
            video_path,
            num_segments=num_segments,
            max_num=max_num_tiles_per_frame,
            get_frame_by_duration=get_frame_by_duration,
            input_size=input_size,
        )

        # move to model device (same as your demo)
        pixel_values = pixel_values.to(_model.device).half()

        # build prefix (same as your demo)
        video_prefix = "".join([f"Frame{i+1}: <image>\n" for i in range(len(num_patches_list))])

        full_prompt = question + video_prefix

        # single-turn chat (same as your demo)
        output, _history = _model.chat(
            _tokenizer,
            pixel_values,
            full_prompt,
            gen_cfg,
            num_patches_list=num_patches_list,
            history=None,
            return_history=True,
        )

        return output