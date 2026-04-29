# this is the working demo + memory & length metrics
import time
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
#from modeling_internvl_model import InternVLChatModel
from modeling_internvl_chat_hico2 import InternVLChatModel

# ------------------ helpers ------------------
def _bytes_to_mb(n: int) -> float:
    return n / (1024.0 ** 2)

def _print_cuda_mem(prefix: str = ""):
    dev = torch.cuda.current_device()
    allocated = torch.cuda.memory_allocated(dev)
    reserved  = torch.cuda.memory_reserved(dev)
    max_alloc = torch.cuda.max_memory_allocated(dev)
    print(f"{prefix}CUDA mem — allocated: {allocated/1e6:.1f} MB | reserved: {reserved/1e6:.1f} MB | peak allocated: {max_alloc/1e6:.1f} MB")

# ------------------ model setting ------------------
model_path = 'OpenGVLab/InternVideo2_5_Chat_8B'
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = InternVLChatModel.from_pretrained(model_path, trust_remote_code=True).half().cuda()
#model = InternVLChatModel.from_pretrained(model_path, trust_remote_code=True).cuda()

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
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
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)
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
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image, input_size=448, max_num=6):
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([int(start_idx + (seg_size / 2) + np.round(seg_size * idx)) for idx in range(num_segments)])
    return frame_indices

def get_num_frames_by_duration(duration):
    local_num_frames = 4
    num_segments = int(duration // local_num_frames)
    if num_segments == 0:
        num_frames = local_num_frames
    else:
        num_frames = local_num_frames * num_segments
    num_frames = min(512, num_frames)
    num_frames = max(128, num_frames)
    return num_frames

def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=0, get_frame_by_duration=False):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    if get_frame_by_duration:
        duration = max_frame / fps
        num_segments = get_num_frames_by_duration(duration)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert("RGB")
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list

# ------------------ evaluation setting ------------------
max_num_frames = 512
generation_config = dict(
    do_sample=False,
    temperature=0.0,
    max_new_tokens=1024,
    top_p=0.1,
    num_beams=1
)
video_path = "expert_10s.mp4"  # concat_expert_novice.mp4 , temporal_concat_video.mp4
num_segments = 8

with torch.no_grad():
    pixel_values, num_patches_list = load_video(
        video_path, num_segments=num_segments, max_num=1, get_frame_by_duration=False
    )
    pixel_values = pixel_values.half().to(model.device)

    frame_count = len(num_patches_list)  # number of sampled frames (each entry corresponds to a frame)
    print(f"[INFO] Frames used: {frame_count}")

    video_prefix = "".join([f"Frame{i+1}: <image>\n" for i in range(frame_count)])

    # ------------------ your question ------------------
    question1 = ("what is emotion emitted by the man in temporal order?")
    question = video_prefix + question1

    # ------------------ token lengths (instruction) ------------------
    ins_tokens = tokenizer(
        question, return_tensors="pt", add_special_tokens=True, truncation=False
    ).input_ids[0]
    instruction_len = int(ins_tokens.numel())
    print(f"[INFO] Instruction length (tokens): {instruction_len}")

    # ------------------ CUDA memory before inference ------------------
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before_alloc = torch.cuda.memory_allocated()
    before_resv  = torch.cuda.memory_reserved()
    t0 = time.time()

    # ------------------ inference ------------------
    output1, chat_history = model.chat(
        tokenizer, pixel_values, question, generation_config,
        num_patches_list=num_patches_list, history=None, return_history=True
    )

    torch.cuda.synchronize()
    t1 = time.time()

    # ------------------ token lengths (output) ------------------
    out_tokens = tokenizer(
        output1, return_tensors="pt", add_special_tokens=True, truncation=False
    ).input_ids[0]
    output_len = int(out_tokens.numel())
    print(f"[INFO] Output length (tokens): {output_len}")

    # ------------------ CUDA memory after inference ------------------
    after_alloc = torch.cuda.memory_allocated()
    after_resv  = torch.cuda.memory_reserved()
    peak_alloc  = torch.cuda.max_memory_allocated()

    delta_peak  = max(0, peak_alloc - before_alloc)
    delta_alloc = max(0, after_alloc - before_alloc)
    delta_resv  = max(0, after_resv  - before_resv)

    print(f"[INFO] Inference time: {(t1 - t0)*1000:.1f} ms")
    print("[MEM] ---------- CUDA Memory Usage (MB) ----------")
    print(f"[MEM] Before  -> allocated: {_bytes_to_mb(before_alloc):.1f} | reserved: {_bytes_to_mb(before_resv):.1f}")
    print(f"[MEM] After   -> allocated: {_bytes_to_mb(after_alloc):.1f}  | reserved: {_bytes_to_mb(after_resv):.1f}")
    print(f"[MEM] Peak    -> peak allocated during chat: {_bytes_to_mb(peak_alloc):.1f}")
    print(f"[MEM] Deltas  -> Δallocated: {_bytes_to_mb(delta_alloc):.1f} | Δreserved: {_bytes_to_mb(delta_resv):.1f} | Δpeak_vs_before: {_bytes_to_mb(delta_peak):.1f}")
    print("-------------------------------------------------")

    # ------------------ final output ------------------
    print("\n===== MODEL OUTPUT =====\n")
    print(output1)
