import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, TextIteratorStreamer
from threading import Thread
from modeling_internvl_chat_hico2 import InternVLChatModel
from conversation import get_conv_template

# =========================
# Model settings
# =========================
model_path = 'OpenGVLab/InternVideo2_5_Chat_8B'
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = InternVLChatModel.from_pretrained(model_path, trust_remote_code=True).cuda()
model.eval()

# =========================
# Preprocessing helpers
# =========================
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

    assert len(processed_images) == blocks

    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)

    return processed_images


def load_image(image, input_size=448, max_num=6):
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(
        image, image_size=input_size, use_thumbnail=True, max_num=max_num
    )
    pixel_values = [transform(img) for img in images]
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
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
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

    pixel_values_list = []
    num_patches_list = []
    transform = build_transform(input_size=input_size)

    if get_frame_by_duration:
        duration = max_frame / fps
        num_segments = get_num_frames_by_duration(duration)

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


# =========================
# Generation config
# =========================
generation_config = {
    "do_sample": False,
    "temperature": 0.0,
    "max_new_tokens": 1024,
    "top_p": 0.1,
    "num_beams": 1,
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
  "openness": 0-1,
  "conscientiousness": 0-1,
  "extraversion": 0-1,
  "agreeableness": 0-1,
  "neuroticism": 0-1,
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
- agent_response must be consider ethnicity, emotion, age,personality
- avatar_emotion must align with the detected emotion
- mixamo_gesture must be one of the allowed values only
"""

def build_question2(text: str):
    safe_text = str(text or "").replace("{", "(").replace("}", ")").strip()

    return f"""
Analyze the video and the user utterance.

User utterance: "{safe_text}"

Your job:
- understand the person's visible emotion
- understand any visible gesture
- estimate a lightweight interaction style from the person's age range, gender presentation, emotion, and Big Five-like personality tendencies
- generate the assistant response style based on those signals

Important:
- Use age range, gender presentation, emotion, and personality only as soft style cues for how to respond
- Do not mention age, gender, or personality in the output
- Do not output analysis
- Do not output markdown
- Return ONLY valid JSON with exactly these 3 keys

Response rules:
- "agent_response" must be short, natural, consider nationality, and suitable for the detected emotion/personality
- "avatar_emotion" must match the emotional tone of the reply
- "mixamo_gesture" must match the person's situation and the assistant's social response

Allowed values:
- avatar_emotion: neutral, happy, sad, angry, fear, disgust, surprised
- mixamo_gesture: Waving, Nod, Thinking, Agreeing, Acknowledging, Clap, None

JSON schema:
{{
  "avatar_emotion": "neutral|happy|sad|angry|fear|disgust|surprised",
  "mixamo_gesture": "Waving|Nod|Thinking|Agreeing|Acknowledging|Clap|None"
  "agent_response": "short empathetic reply",
  
}}
""".strip()

# =========================
# Input
# =========================
video_path = "uploads/derived/ws_dhanuja_session_noaudio.mp4"
num_segments = 16
user_utterance = "How are you today? "

#prompt = f"""You are a multimodal AI assistant for the SELA system. Analyze the given video together with this user utterance: {repr(user_utterance)}. Return the result as a valid CSV table with exactly 2 lines only: (1) one header row and (2) one data row. Do not output key-value pairs. Do not put one field per line. Use exactly these columns in this exact order: utterance,age_range,gender,emotion,openness,conscientiousness,extraversion,agreeableness,neuroticism,language_or_cultural_cue,social_signals,agent_response,avatar_emotion,mixamo_gesture. The fields utterance, age_range, gender, emotion, language_or_cultural_cue, social_signals, agent_response, avatar_emotion, and mixamo_gesture must be text in double quotes. The fields openness, conscientiousness, extraversion, agreeableness, and neuroticism must be numeric values between 0 and 1. social_signals must be one single quoted field containing 3 to 5 short comma-separated signals. agent_response must be one short natural sentence. avatar_emotion must be suitable for the avatar face. mixamo_gesture must be a valid short gesture label such as "Idle", "Talking", "Nod", or "Waving". Output only the CSV header and one CSV data row, nothing else."""
prompt = build_question2("what is the capital of my country")
# =========================
# Run
# =========================
with torch.no_grad():
    pixel_values, num_patches_list = load_video(
        video_path,
        num_segments=num_segments,
        max_num=1,
        get_frame_by_duration=False
    )
    pixel_values = pixel_values.to(model.device)

    print("pixel_values shape:", pixel_values.shape)
    print("num_patches_list:", num_patches_list)

    # One <image> placeholder per frame
    video_prefix = "".join([f"Frame{i+1}: <image>\n" for i in range(len(num_patches_list))])

    # Final question text
    question = prompt + "\n" + video_prefix

    # Build conversation prompt
    template = get_conv_template(model.template)
    template.system_message = model.system_message
    template.append_message(template.roles[0], question)
    template.append_message(template.roles[1], None)
    query = template.get_prompt()

    # Required special tokens
    IMG_START_TOKEN = "<img>"
    IMG_END_TOKEN = "</img>"
    IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"

    # Required for model.generate()
    model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)

    # Replace each <image> with image token block
    for num_patches in num_patches_list:
        image_tokens = (
            IMG_START_TOKEN
            + IMG_CONTEXT_TOKEN * model.num_image_token * num_patches
            + IMG_END_TOKEN
        )
        query = query.replace("<image>", image_tokens, 1)

    # Tokenize FULL query
    model_inputs = tokenizer(query, return_tensors="pt")
    input_ids = model_inputs["input_ids"].to(model.device)
    attention_mask = model_inputs["attention_mask"].to(model.device)

    # Same eos handling as chat()
    eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())
    generation_config["eos_token_id"] = eos_token_id

    # Stream token by token
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
        **generation_config
    )

    print("\n===== MODEL OUTPUT =====\n")

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    response = ""
    for new_text in streamer:
        print(new_text, end="", flush=True)
        response += new_text

    thread.join()
    print()

    response = response.split(template.sep.strip())[0].strip()
    #print("\n===== FINAL CLEANED RESPONSE =====\n")
    #print(response)