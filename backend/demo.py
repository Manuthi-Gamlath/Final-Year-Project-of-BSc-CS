# this is the working demo
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
#from modeling_internvl_model import InternVLChatModel
from modeling_internvl_chat_hico2 import InternVLChatModel

# model setting
model_path = 'OpenGVLab/InternVideo2_5_Chat_8B'
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
#model = InternVLChatModel.from_pretrained(model_path, trust_remote_code=True).half().cuda()
model = InternVLChatModel.from_pretrained(model_path, trust_remote_code=True).cuda()


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img), T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC), T.ToTensor(), T.Normalize(mean=MEAN, std=STD)])
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
    target_ratios = set((i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = ((i % (target_width // image_size)) * image_size, (i // (target_width // image_size)) * image_size, ((i % (target_width // image_size)) + 1) * image_size, ((i // (target_width // image_size)) + 1) * image_size)
        # split the image
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

def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=0, get_frame_by_duration = False):
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

# evaluation setting


max_num_frames = 512
generation_config = dict(
    do_sample=False,
    temperature=0.0,
    max_new_tokens=1024,
    top_p=0.1,
    num_beams=1
)
video_path = "uploads/derived/ws_dhanuja_session_noaudio.mp4" #concat_expert_novice.mp4 ,  temporal_concat_video.mp4
#video_path = "expert_video_0.mp4"
num_segments=16


with torch.no_grad():
  
  pixel_values, num_patches_list = load_video(video_path, num_segments=num_segments, max_num=1, get_frame_by_duration=False)
  #pixel_values = pixel_values.half().to(model.device)
  pixel_values = pixel_values.to(model.device)
  print('sssssssssssssssssssssssss',(pixel_values.shape),model.device)
  video_prefix = "".join([f"Frame{i+1}: <image>\n" for i in range(len(num_patches_list))])
  #aa=model.forward(pixel_values)
  #print(video_prefix)
  # single-turn conversation
  #question1 = "give engaument level  of the person per frame?"
  #question1 = "expert and novice are interacting in provided two video segments concataneated as one video, Eight frames from each segment are sampled from 61 frames based on social signals of both expert and novice.. describe the video in details?"
  #question1 = "In the provided video, the expert and novice are communicating via a screen,Given this description, are they interacting? "
  #question1 = "how many people are there in the video?"
  #question1 = (
  #  "For the person in the video Describe the person's head movement in temporal order:\n"
  # # "3. Describe the person's head movement in short words from: NODDING, SHAKING, TILTING, STILL, TURNING LEFT, TURNING RIGHT, LOOKING DOWN, LOOKING UP, RANDOM MOVEMENT.\n"
  #)
  #question1 = 'left person is expert and right person is novice,describe the both person\'s hand gesture,base on that what is the engagement value of expert for each frame?'
  #question1 = 'left person is expert and right person is novice,describe the both person\'s hand gestures'
  #question1 = 'left person is expert and right person is novice,they are having a conversation via tele comferance,provide engagemnt level as a value from 0-1 for the experts based on  Head Direction movemnt only in tempotal order using Forward,Back,Upwards,Downwards,Sideways'
  #question1 = 'left person is expert and right person is novice,they are having a conversation via tele comferance,provide engagemnt level as a value from 0-1 for the experts based on head only in temporal order'
  #question1 = "Watch this stitched video segment (expert-left, novice-right),they are having a conversation via tele comferance, Detect (1)hand gestures (2)head movement (3)Facial Expressions (5)Gaze direction (6)Hand Rest Positions and infer engagement scores for expert(0–1) in every frame"
  #question1 = "Watch this stitched video segment,left person is expert and right person is novice,they are having a conversation via tele comferance, Detect (1)hand gestures (2)head movement (3)Facial Expressions (5)Gaze direction (6)Hand Rest Positions and infer engagement scores for expert(0–1)"
  
  
  #question1 = "Watch this stitched video segment,left person is expert and right person is novice,they are having a conversation via tele comferance, what is experts's (1) Gaze direction from answers only UP,TOWARDS INTERLOCUTOR , DOWN , SIDEWAYS , OTHER for all 16 frame?"
  #question1 = "Watch this stitched video segment. The left person is the expert and the right person is the novice; they are having a conversation via teleconference. What are the expert’s eyebrow movements (FROWN or RAISED) for all 16 frames?"
  #question1 = "Watch this stitched video segment. The left person is the expert and the right person is the novice; they are having a conversation via teleconference. What are the expert’s gestures (choose only from ICONIC, METAPHORIC, DEICTIC, BEAT, ADAPTOR) for all 16 frames?"
  
#  question1= (
#    "For the person in the video:\n"
#    "1. Is the person looking at the screen? (Answer: Yes/No)\n"
#    "2. Is the person moving mouth? (Answer: Yes/No)\n"
#    "3. Describe the person's head movement in short words from: NODDING, SHAKING, TILTING, STILL, TURNING LEFT, TURNING RIGHT, LOOKING DOWN, LOOKING UP, RANDOM MOVEMENT.\n"
#    "4. Describe the hand gesture in one word from: ARMS CROSSED, HANDS TOGETHER, HANDS IN POCKETS, HANDS BEHIND BACK, AKIMBO, Random Movement, No Movement.\n"
#    "5. Describe the body pose in one word from: LEANING FORWARD, LEANING BACK, RELAXED, TENSE, WALKING, RUNNING, TURNED SIDEWAYS, LYING DOWN, CROUCHING.\n"
#    "6. Describe the person's emotion in one word from: Happiness, Sadness, Fear, Anger, Surprise, Disgust, Contempt, or Neutral.\n"
#    "7. Describe the person's Gaze direction in one word from: straight ahead , UP , DOWN , SIDEWAYS , OTHER.\n"
#)
  #question1 = "Describe the only gaze direction  of the man in temporal order?"
  #question1 = "Describe the only Head Movement  of the man in temporal order?"
  #question1 = "Is the man smiling?" 
  #question1 = "Describe only the hand gesture of the woman?"
  #question1 = "Describe only the hand rest position of the woman?"
  #question1 = "what is value that a participant in an interaction attributes to the goal of being together with the other participant(s) and of continuing the interaction?"
 
  #question1 = "You are a multimodal AI assistant for the SELA (Socially-aware Embodied Language Agent) system; analyze the given video and infer the person’s estimated age range, detected emotion (e.g., happy, sad, neutral, stressed), emotion confidence (0–1), engagement level (low, medium, high), key observable social signals (such as facial expression, head movement, gaze, and speech tone), and generate a short natural conversational response; output the result in strict CSV format only using the schema age_range,emotion,emotion_confidence,engagement_level,social_signals,agent_response, with no explanations, no markdown, and no extra text, ensuring social signals are concise and comma-separated within quotes and the response is one short natural sentence."
 # question1 = "You are a multimodal AI assistant for the SELA (Socially-aware Embodied Language Agent) system; analyze the given video and infer the person’s estimated age range, detected emotion (e.g., happy, sad, neutral, stressed), emotion confidence (0–1), engagement level (low, medium, high), tentative OCEAN-style personality cues as best-effort estimates only, key observable social signals (such as facial expression, head movement, gaze, posture, and speech tone), and generate a short natural conversational response; do not guess or infer nationality from appearance alone, and only use nationality, culture, language, or accent if it is explicitly stated or clearly expressed in the audio; tailor the response using the detected emotion, estimated age range, and any explicit language/cultural cues; also output the recommended avatar emotion and the most suitable Mixamo body gesture for the response; output the result in strict CSV format only using the schema age_range,emotion,emotion_confidence,engagement_level,openness,conscientiousness,extraversion,agreeableness,neuroticism,language_or_cultural_cue,social_signals,agent_response,avatar_emotion,mixamo_gesture, with no explanations, no markdown, and no extra text, ensuring social_signals are concise and comma-separated within quotes, OCEAN values are low/medium/high, language_or_cultural_cue is filled only when explicitly supported by speech, and the response is one short natural sentence."
  user_utterance = "How are you today?"

  prompt = f"""You are a multimodal AI assistant for the SELA system. Analyze the given video together with this user utterance: {repr(user_utterance)}. Return the result as a valid CSV table with exactly 2 lines only: (1) one header row and (2) one data row. Do not output key-value pairs. Do not put one field per line. Use exactly these columns in this exact order: utterance,age_range,gender,emotion,openness,conscientiousness,extraversion,agreeableness,neuroticism,language_or_cultural_cue,social_signals,agent_response,avatar_emotion,mixamo_gesture. The fields utterance, age_range, gender, emotion, language_or_cultural_cue, social_signals, agent_response, avatar_emotion, and mixamo_gesture must be text in double quotes. The fields openness, conscientiousness, extraversion, agreeableness, and neuroticism must be numeric values between 0 and 1. social_signals must be one single quoted field containing 3 to 5 short comma-separated signals. agent_response must be one short natural sentence. avatar_emotion must be suitable for the avatar face. mixamo_gesture must be a valid short gesture label such as "Idle", "Talking", "Nod", or "Waving". Output only the CSV header and one CSV data row, nothing else."""
  
  ####question1 = 'left person is expert and right person is novice,they are having a conversation via tele comferance,based on the both person\'s (1)hand gestures (2)head movement (3)body pose (5)Gaze direction, what are the engagements of expert quntified between 0-1?'
  ####question1 = 'left person is expert and right person is novice,describe the both person\'s head movement,,are they having?'
  ####question1=    "left person is expert and right person is novice,they are having a conversation via tele comferance.are they interacting? and why do you come to that answer?"
  #question1=    "left person is expert and right person is novice,(1) are they interacting? (2) if no,explain the why do you think no"
  
  #question1 = "Describe the head movement in one word from: NODDING, SHAKING, TILTING, STILL, TURNING LEFT, TURNING RIGHT, LOOKING DOWN, LOOKING UP, RANDOM MOVEMENT.\n"
  question = prompt+video_prefix  
  #print("----------------------------------------")
  output1, chat_history = model.chat(tokenizer, pixel_values, question, generation_config, num_patches_list=num_patches_list, history=None, return_history=True)
  print(output1)
  
  ## multi-turn conversation

  #question2 = "How many people appear in the video?"
  #output2, chat_history = model.chat(tokenizer, pixel_values, question, generation_config, num_patches_list=num_patches_list, history=chat_history, return_history=True)
  
  #print(output2)[p]

#  def score_from_cues(cues):
#    # cues is a dict with keys like:
#    # 'gaze', 'is_speaking', 'head_movement', 'head_direction',
#    # 'smile', 'eyebrow', 'gesture', 'hand_rest',
#    # 'pitch_norm', 'intensity_norm'
#
#    # ---- Gaze ----
#    gaze = cues.get('gaze', 'OTHER')
#    if gaze == 'TOWARDS_INTERLOCUTOR':
#        gaze_score = 1.0
#    elif gaze == 'SIDEWAYS':
#        gaze_score = 0.4
#    elif gaze in ['UP', 'DOWN']:
#        gaze_score = 0.3
#    else:
#        gaze_score = 0.2
#
#    # ---- Speaking ----
#    speaking_score = 1.0 if cues.get('is_speaking', False) else 0.3
#
#    # ---- Prosody ----
#    pitch_norm = cues.get('pitch_norm', 0.5)
#    intensity_norm = cues.get('intensity_norm', 0.5)
#    prosody_score = 1.0 - (abs(pitch_norm - 0.6) + abs(intensity_norm - 0.6)) / 2.0
#    prosody_score = max(0.0, min(1.0, prosody_score))
#
#    # ---- Head movement ----
#    hm = cues.get('head_movement', None)
#    if hm == 'NOD':
#        nod_score = 1.0
#    elif hm == 'SHAKE':
#        nod_score = 0.4
#    else:
#        nod_score = 0.3
#
#    # ---- Head direction ----
#    hd = cues.get('head_direction', None)
#    if hd in ['FORWARD', 'SIDE_TILT']:
#        head_dir_score = 0.8
#    elif hd in ['UPWARDS', 'DOWNWARDS']:
#        head_dir_score = 0.5
#    elif hd == 'SIDEWAYS':
#        head_dir_score = 0.4
#    elif hd == 'BACK':
#        head_dir_score = 0.3
#    else:
#        head_dir_score = 0.5
#
#    # ---- Smile ----
#    smile_score = 0.8 if cues.get('smile') == 'SMILE' else 0.4
#
#    # ---- Eyebrows ----
#    eb = cues.get('eyebrow', 'NO_MOVEMENT')
#    if eb == 'RAISED':
#        brow_score = 0.7
#    elif eb == 'FROWN':
#        brow_score = 0.4
#    else:
#        brow_score = 0.5
#
#    # ---- Gesture ----
#    gest = cues.get('gesture', 'NONE')
#    if gest in ['ICONIC', 'METAPHORIC', 'DEICTIC']:
#        gesture_score = 1.0
#    elif gest == 'BEAT':
#        gesture_score = 0.8
#    elif gest == 'ADAPTOR':
#        gesture_score = 0.4
#    else:
#        gesture_score = 0.3
#
#    # ---- Hand rest ----
#    hr = cues.get('hand_rest', None)
#    if hr == 'NO_REST':
#        hand_score = 0.8
#    elif hr == 'HANDS_TOGETHER':
#        hand_score = 0.6
#    elif hr in ['AKIMBO', 'HANDS_BEHIND_BACK']:
#        hand_score = 0.5
#    elif hr in ['ARMS_CROSSED', 'HANDS_IN_POCKETS']:
#        hand_score = 0.3
#    else:
#        hand_score = 0.5
#
#    # ---- Weighted sum ----
#    E_raw = (
#        0.20 * gaze_score +
#        0.20 * speaking_score +
#        0.15 * prosody_score +
#        0.15 * gesture_score +
#        0.10 * nod_score +
#        0.07 * smile_score +
#        0.05 * head_dir_score +
#        0.04 * hand_score +
#        0.04 * brow_score
#    )
#
#    # Clamp
#    E = max(0.0, min(1.0, E_raw))
#    return E
