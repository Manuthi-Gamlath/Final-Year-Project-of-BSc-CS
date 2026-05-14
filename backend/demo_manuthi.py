# ===== Model Compression Toolkit for InternVLChatModel =====
# Methods: INT8/INT4 (bitsandbytes), Pruning (magnitude/structured), Low-rank SVD
# Keeps your AMP inference friendly; prints a compression summary.
# ----------------------------------------------------------

import os, re, math, time, importlib, numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torchvision.transforms as T
from PIL import Image
from decord import VideoReader, cpu
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer

# --- your model class ---
from modeling_internvl_chat_hico2 import InternVLChatModel

# ------------------ USER KNOBS ------------------
MODEL_PATH = "OpenGVLab/InternVideo2_5_Chat_8B"
VIDEO_PATH = "expert_video_0.mp4"
NUM_SEGMENTS = 8
MAX_NEW_TOKENS = 512

# Choose the compression pipeline (set True/False)
USE_INT8 = False
USE_INT4 = True            # set True for 4-bit (NF4); requires bitsandbytes
USE_PRUNING = True        # magnitude/structured pruning
USE_LOWRANK = False        # SVD low-rank factorization

# Quantization settings
INT4_QUANT_TYPE = "nf4"    # "nf4" or "fp4"
INT4_DOUBLE_QUANT = True
BNB_COMPUTE_DTYPE = torch.float16

# Pruning settings
PRUNE_GLOBAL_AMOUNT = 0.2  # 20% global unstructured pruning (0..1)
PRUNE_STRUCTURED = True   # if True, prune entire neurons (out_features) per Linear
PRUNE_STRUCTURED_AMOUNT = 0.15  # 15% channels per matched Linear
PRUNE_INCLUDE = (r"language", r"\bllm\b", r"transformer", r"mlp", r"fc", r"proj", r"lm_head")
PRUNE_EXCLUDE = (r"vision", r"visual")

# Low-rank settings
LOWRANK_RATIO = 0.5        # keep rank = int(min(in,out)*LOWRANK_RATIO)
LOWRANK_INCLUDE = PRUNE_INCLUDE
LOWRANK_EXCLUDE = PRUNE_EXCLUDE

# Inference dtype (activations) after compression
COMPUTE_DTYPE_FALLBACK = torch.float16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# modest speedup on Ampere/Hopper
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("medium")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ------------------ UTILS ------------------
def _bytes_to_mb(n): return n/(1024.0**2)

def _compile_regex(ps): return [re.compile(p, re.IGNORECASE) for p in ps] if ps else []

def _match(name, inc, exc):
    if inc and not any(p.search(name) for p in inc): return False
    if exc and any(p.search(name) for p in exc): return False
    return True

def build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    ow, oh = image.size
    target = sorted({(i, j) for n in range(min_num, max_num+1)
                     for i in range(1, n+1) for j in range(1, n+1)
                     if 1 <= i*j <= max_num}, key=lambda x: x[0]*x[1])
    ar = ow/oh
    best = min(target, key=lambda r: abs(ar - (r[0]/r[1])))
    tw, th = image_size*best[0], image_size*best[1]
    blocks = best[0]*best[1]
    resized = image.resize((tw, th))
    out = []
    for k in range(blocks):
        gx = (k % (tw//image_size))*image_size
        gy = (k // (tw//image_size))*image_size
        out.append(resized.crop((gx, gy, gx+image_size, gy+image_size)))
    if use_thumbnail and len(out) != 1:
        out.append(image.resize((image_size, image_size)))
    return out

def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound: start, end = bound
    else: start, end = -100000, 100000
    start_idx = max(first_idx, round(start*fps))
    end_idx   = min(round(end*fps), max_frame)
    seg = float(end_idx - start_idx)/num_segments
    return np.array([int(start_idx + (seg/2) + np.round(seg*idx)) for idx in range(num_segments)])

def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=0, get_frame_by_duration=False):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())
    if get_frame_by_duration:
        duration = max_frame/fps
        nf = max(128, min(512, (int(duration//4) or 1)*4))
        num_segments = nf
    idxs = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    tr = build_transform(448)
    pvals, npatches = [], []
    for i in idxs:
        tiles = dynamic_preprocess(Image.fromarray(vr[i].asnumpy()).convert("RGB"),
                                   image_size=448, use_thumbnail=True, max_num=max_num)
        pv = torch.stack([tr(t) for t in tiles])
        pvals.append(pv); npatches.append(pv.shape[0])
    return torch.cat(pvals), npatches

def parameter_dtype_hist(model):
    d = {}
    for _, p in model.named_parameters():
        d[str(p.dtype)] = d.get(str(p.dtype), 0) + p.numel()
    return d

# ------------------ QUANTIZATION (bitsandbytes) ------------------
def try_intx_quantized_load(int_mode="int4"):
    """
    Return (model, tokenizer, used_bnb:bool, compute_dtype)
    int_mode: "int8" or "int4"
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    # bitsandbytes availability
    try:
        import bitsandbytes as bnb  # noqa
        from transformers import BitsAndBytesConfig
    except Exception as e:
        print(f"[Q] bitsandbytes not available: {e}")
        return None, tokenizer, False, COMPUTE_DTYPE_FALLBACK

    if int_mode == "int8":
        bcfg = BitsAndBytesConfig(load_in_8bit=True)
    else:
        bcfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=INT4_QUANT_TYPE,
            bnb_4bit_use_double_quant=INT4_DOUBLE_QUANT,
            bnb_4bit_compute_dtype=BNB_COMPUTE_DTYPE
        )

    try:
        model = InternVLChatModel.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            device_map="auto",
            quantization_config=bcfg,
            torch_dtype=BNB_COMPUTE_DTYPE
        )
        model.eval()
        print(f"[Q] Loaded {int_mode.upper()} quantized model via HF + bitsandbytes.")
        return model, tokenizer, True, BNB_COMPUTE_DTYPE
    except TypeError as e:
        print("[Q] Custom model rejected quantization_config; will try dense + manual swaps. Error:", e)
    except Exception as e:
        print("[Q] Quantized load failed:", e)

    return None, tokenizer, False, COMPUTE_DTYPE_FALLBACK

def apply_manual_int4_swap(model,
                           include=PRUNE_INCLUDE,
                           exclude=PRUNE_EXCLUDE,
                           compute_dtype=BNB_COMPUTE_DTYPE,
                           quant_type=INT4_QUANT_TYPE,
                           use_double_quant=INT4_DOUBLE_QUANT):
    """Swap nn.Linear -> bnb.nn.Linear4bit for included modules."""
    try:
        bnb_nn = importlib.import_module("bitsandbytes.nn")
    except Exception as e:
        raise RuntimeError("bitsandbytes is required for manual INT4 swapping") from e

    Linear4bit = getattr(bnb_nn, "Linear4bit", None)
    if Linear4bit is None:
        raise RuntimeError("bitsandbytes.nn.Linear4bit not found")

    inc = _compile_regex(include)
    exc = _compile_regex(exclude)
    replaced = 0

    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if isinstance(child, nn.Linear) and _match(full, inc, exc):
                new_lin = Linear4bit(
                    child.in_features, child.out_features,
                    bias=(child.bias is not None),
                    compute_dtype=compute_dtype,
                    quant_type=quant_type,
                    compress_statistics=use_double_quant
                )
                with torch.no_grad():
                    new_lin.weight.data = child.weight.data.clone()
                    if child.bias is not None:
                        new_lin.bias.data = child.bias.data.clone()
                setattr(module, child_name, new_lin)
                replaced += 1

    print(f"[Q-MANUAL] Replaced {replaced} Linear layers with Linear4bit ({quant_type}, double={use_double_quant}).")
    return replaced

# ------------------ PRUNING ------------------
def apply_global_magnitude_pruning(model, amount=0.2, include=PRUNE_INCLUDE, exclude=PRUNE_EXCLUDE):
    """Global unstructured pruning over matched Linear weights."""
    inc = _compile_regex(include); exc = _compile_regex(exclude)
    params_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and _match(name, inc, exc):
            params_to_prune.append((module, "weight"))
    if not params_to_prune:
        print("[PRUNE] No matched Linear layers for global pruning.")
        return 0

    prune.global_unstructured(params_to_prune, pruning_method=prune.L1Unstructured, amount=amount)
    for m, _ in params_to_prune:
        prune.remove(m, "weight")
    print(f"[PRUNE] Applied global L1 unstructured pruning: amount={amount}, layers={len(params_to_prune)}")
    return len(params_to_prune)

def apply_structured_neuron_pruning(model, amount=0.15, include=PRUNE_INCLUDE, exclude=PRUNE_EXCLUDE):
    """Structured pruning: remove entire output neurons (channels) per Linear."""
    inc = _compile_regex(include); exc = _compile_regex(exclude)
    count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and _match(name, inc, exc):
            prune.ln_structured(module, name="weight", amount=amount, n=2, dim=0)  # prune out_features by L2 norm
            prune.remove(module, "weight")
            count += 1
    print(f"[PRUNE] Applied structured neuron pruning on {count} Linear layers (amount={amount}).")
    return count

# ------------------ LOW-RANK (SVD) ------------------
class LowRankLinear(nn.Module):
    """Two-layer factorization: W[mxn] ~ (A[mxr] @ B[rxn]); bias kept."""
    def __init__(self, in_f, out_f, rank, bias=True):
        super().__init__()
        self.A = nn.Linear(in_f, rank, bias=False)
        self.B = nn.Linear(rank, out_f, bias=bias)
    def forward(self, x):  # (N, *, in_f)
        return self.B(self.A(x))

def apply_lowrank_factorization(model, ratio=0.5, include=LOWRANK_INCLUDE, exclude=LOWRANK_EXCLUDE):
    inc = _compile_regex(include); exc = _compile_regex(exclude)
    replaced = 0
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if isinstance(child, nn.Linear) and _match(full, inc, exc):
                in_f, out_f = child.in_features, child.out_features
                rank = max(1, int(min(in_f, out_f) * ratio))
                # SVD on CPU to save GPU mem
                W = child.weight.data.float().cpu()  # [out_f, in_f]
                try:
                    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
                except RuntimeError:
                    # fallback to truncated via eigh on W@W^T
                    M = W @ W.T
                    evals, evecs = torch.linalg.eigh(M)
                    idx = torch.argsort(evals, descending=True)
                    U = evecs[:, idx][:, :rank]
                    S = torch.sqrt(torch.clamp(evals[idx][:rank], min=0))
                    Vh = (U.T @ W)

                U_r = U[:, :rank]
                S_r = S[:rank]
                Vh_r = Vh[:rank, :]

                A_w = (Vh_r.T @ torch.diag(S_r))  # [in_f, rank]
                B_w = U_r                          # [out_f, rank]

                new_lin = LowRankLinear(in_f, out_f, rank, bias=(child.bias is not None))
                with torch.no_grad():
                    new_lin.A.weight.copy_(A_w.T)                      # [rank, in_f]
                    new_lin.B.weight.copy_(B_w)                         # [out_f, rank]
                    if child.bias is not None:
                        new_lin.B.bias.copy_(child.bias.data)
                setattr(module, child_name, new_lin)
                replaced += 1
    print(f"[LRANK] Replaced {replaced} Linear layers with LowRankLinear (ratio={ratio}).")
    return replaced

# ------------------ COMPRESSION SUMMARY ------------------
def print_compression_summary(model, tag=""):
    print(f"\n===== COMPRESSION SUMMARY {tag} =====")
    # Try to detect bnb wrappers
    quant_present = False
    try:
        bnb_nn = importlib.import_module("bitsandbytes.nn")
        wrappers = tuple(w for w in (getattr(bnb_nn, "Linear4bit", None),
                                     getattr(bnb_nn, "Linear8bitLt", None)) if w)
        for _, m in model.named_modules():
            if wrappers and isinstance(m, wrappers):
                quant_present = True; break
    except Exception:
        pass
    print("Quant wrappers present:", quant_present)

    # Count LowRankLinear
    lr_count = sum(1 for _, m in model.named_modules() if isinstance(m, LowRankLinear))
    print("LowRankLinear layers:", lr_count)

    # Param dtype histogram
    dth = parameter_dtype_hist(model)
    print("Parameter dtype histogram:", dth)

    # Rough param memory estimate (dense params only)
    def _sz(dt):
        if "float32" in dt: return 4
        if "float16" in dt or "bfloat16" in dt: return 2
        return None
    total = 0; unknown=False
    for dt, cnt in dth.items():
        sz = _sz(dt)
        if sz is None: unknown=True
        else: total += cnt*sz
    if unknown:
        print("Estimated param memory: (inexact; quantized/packed layers not summed)")
    else:
        print(f"Estimated param memory: {_bytes_to_mb(total):.1f} MB")
    print("=====================================\n")

# ------------------ LOAD & COMPRESS ------------------
def load_model_dense(dtype=COMPUTE_DTYPE_FALLBACK):
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    mdl = InternVLChatModel.from_pretrained(MODEL_PATH, trust_remote_code=True)
    mdl = mdl.to(device=DEVICE, dtype=dtype).eval()
    return mdl, tok

def compress_model_pipeline():
    compute_dtype = COMPUTE_DTYPE_FALLBACK
    used_bnb = False

    # 1) Try bnb quantization first (preferred)
    model, tokenizer = None, None
    if USE_INT4 or USE_INT8:
        int_mode = "int4" if USE_INT4 else "int8"
        model, tokenizer, used_bnb, compute_dtype = try_intx_quantized_load(int_mode=int_mode)

    # 2) Fallback to dense + manual compression
    if model is None:
        model, tokenizer = load_model_dense(dtype=compute_dtype)

        # manual 4-bit swap if requested
        if USE_INT4:
            replaced = apply_manual_int4_swap(model, compute_dtype=BNB_COMPUTE_DTYPE,
                                              quant_type=INT4_QUANT_TYPE, use_double_quant=INT4_DOUBLE_QUANT)
            if replaced == 0:
                print("[Q-MANUAL] No layers replaced; adjust include/exclude patterns.")

        # INT8 manual swap could be added similarly using bnb.nn.Linear8bitLt if needed.

    # 3) Pruning (optional)
    if USE_PRUNING:
        if PRUNE_STRUCTURED:
            apply_structured_neuron_pruning(model, amount=PRUNE_STRUCTURED_AMOUNT)
        else:
            apply_global_magnitude_pruning(model, amount=PRUNE_GLOBAL_AMOUNT)

    # 4) Low-rank factorization (optional)
    if USE_LOWRANK:
        apply_lowrank_factorization(model, ratio=LOWRANK_RATIO)

    # Move to CUDA (don’t change dtype of quantized layers)
    target_device = next(model.parameters()).device
    if target_device.type != "cuda" and torch.cuda.is_available():
        model.to("cuda")

    print_compression_summary(model, tag="(post-compress)")
    return model, tokenizer, compute_dtype

# ------------------ INFERENCE (kept simple) ------------------
def run_demo():
    model, tokenizer, compute_dtype = compress_model_pipeline()

    generation_config = dict(
        do_sample=False, temperature=0.0,
        max_new_tokens=MAX_NEW_TOKENS,
        top_p=0.1, num_beams=1
    )

    with torch.inference_mode():
        pixel_values, num_patches_list = load_video(VIDEO_PATH, num_segments=NUM_SEGMENTS, max_num=1, get_frame_by_duration=False)
        target_device = next(model.parameters()).device
        pixel_values = pixel_values.to(device=target_device, dtype=compute_dtype)

        frame_count = len(num_patches_list)
        print(f"[INFO] Frames used: {frame_count}")
        prefix = "".join([f"Frame{i+1}: <image>\n" for i in range(frame_count)])
        question1 = ("Hello, I am Dhanuja. I am Sri Lankan, I am happy, and I have an introverted personality. "
                     "Please introduce yourself and adapt your response considering my nationality, current emotion, and personality traits.")
        question = prefix + question1

        ins_tokens = tokenizer(question, return_tensors="pt", add_special_tokens=True, truncation=False).input_ids[0]
        print(f"[INFO] Instruction length (tokens): {int(ins_tokens.numel())}")

        if target_device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            before_alloc = torch.cuda.memory_allocated()
            before_resv  = torch.cuda.memory_reserved()

        t0 = time.time()
        from torch import amp
        ctx = amp.autocast(device_type="cuda", dtype=compute_dtype) if target_device.type == "cuda" \
              else amp.autocast(device_type="cpu",  dtype=compute_dtype)

        with ctx:
            output, history = model.chat(
                tokenizer, pixel_values, question, generation_config,
                num_patches_list=num_patches_list, history=None, return_history=True
            )

        if target_device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()

        out_tokens = tokenizer(output, return_tensors="pt", add_special_tokens=True, truncation=False).input_ids[0]
        print(f"[INFO] Output length (tokens): {int(out_tokens.numel())}")

        if target_device.type == "cuda":
            after_alloc = torch.cuda.memory_allocated()
            after_resv  = torch.cuda.memory_reserved()
            peak_alloc  = torch.cuda.max_memory_allocated()
            print(f"[INFO] Inference time: {(t1 - t0)*1000:.1f} ms")
            print("[MEM] ---------- CUDA Memory Usage (MB) ----------")
            print(f"[MEM] Before  -> allocated: {_bytes_to_mb(before_alloc):.1f} | reserved: {_bytes_to_mb(before_resv):.1f}")
            print(f"[MEM] After   -> allocated: {_bytes_to_mb(after_alloc):.1f}  | reserved: {_bytes_to_mb(after_resv):.1f}")
            print(f"[MEM] Peak    -> peak allocated: {_bytes_to_mb(peak_alloc):.1f}")
            print(f"[MEM] Deltas  -> Δalloc: {_bytes_to_mb(after_alloc-before_alloc):.1f} | Δresv: {_bytes_to_mb(after_resv-before_resv):.1f} | Δpeak: {_bytes_to_mb(peak_alloc-before_alloc):.1f}")
            print("-------------------------------------------------")

        print("\n===== MODEL OUTPUT =====\n")
        print(output)

if __name__ == "__main__":
    run_demo()
