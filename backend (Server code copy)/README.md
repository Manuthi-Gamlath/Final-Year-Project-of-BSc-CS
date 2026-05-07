---
language:
- en
library_name: transformers
license: apache-2.0
metrics:
- accuracy
tags:
- multimodal
pipeline_tag: video-text-to-text
model-index:
- name: InternVideo2.5
  results:
  - task:
      type: multimodal
    dataset:
      name: MLVU
      type: mlvu
    metrics:
    - type: accuracy
      value: 72.8
      name: accuracy
      verified: true
  - task:
      type: multimodal
    dataset:
      name: MVBench
      type: mvbench
    metrics:
    - type: accuracy
      value: 75.7
      name: accuracy
      verified: true
  - task:
      type: multimodal
    dataset:
      name: Perception Test
      type: percepTest
    metrics:
    - type: accuracy
      value: 74.9
      name: accuracy
      verified: true
  - task:
      type: multimodal
    dataset:
      name: LongVideoBench
      type: longvideobench
    metrics:
    - type: accuracy
      value: 60.6
      name: accuracy
      verified: true
  - task:
      type: multimodal
    dataset:
      name: VideoMME (w/o sub)
      type: videomme
    metrics:
    - type: accuracy
      value: 65.1
      name: accuracy
      verified: true
  - task:
      type: multimodal
    dataset:
      name: LVBench
      type: lvbench
    metrics:
    - type: accuracy
      value: 46.4
      name: accuracy
      verified: true


---

# 📕InternVideo2.5⚡
<!-- [\[📰 Blog\]](https://internvideo.github.io/blog/2024-12-31-VideoChat-Flash) -->
[\[📂 GitHub\]](https://github.com/OpenGVLab/InternVideo/tree/main/InternVideo2.5)  
[\[📜 Tech Report\]](https://arxiv.org/abs/2501.12386) 
<!-- [\[🗨️ Chat Demo\]](https://huggingface.co/spaces/OpenGVLab/VideoChat-Flash) -->

 InternVideo2.5 is a video multimodal large language model (MLLM, built upoon InternVL2.5) enhanced with **long and rich context (LRC) modeling**. It significantly improves upon existing MLLMs by enhancing their ability to perceive fine-grained details and capture long-form temporal structures. We achieve this through dense vision task annotations using direct preference optimization (TPO) and compact spatiotemporal representations via adaptive hierarchical token compression (HiCo).




## 📈 Performance
| Model |  MVBench | LongVideoBench |  VideoMME(w/o sub)| 
| ---   |  ---     |   ---            | ---     | 
|InternVideo2.5| 75.7 |  60.6   | 65.1| 

## 🚀 How to use the model

First, you need to install [flash attention2](https://github.com/Dao-AILab/flash-attention) and some other modules. We provide a simple installation example below:
```
pip install transformers==4.40.1
pip install av
pip install imageio
pip install decord
pip install opencv-python
pip install flash-attn --no-build-isolation
```
Then you could use our model:
```python
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer


# model setting
model_path = 'OpenGVLab/InternVideo2_5_Chat_8B'

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda()

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

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

def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32, get_frame_by_duration = False):
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
video_path = "your_video.mp4"
num_segments=128


with torch.no_grad():
  
  pixel_values, num_patches_list = load_video(video_path, num_segments=num_segments, max_num=1, get_frame_by_duration=False)
  pixel_values = pixel_values.to(torch.bfloat16).to(model.device)
  video_prefix = "".join([f"Frame{i+1}: <image>\n" for i in range(len(num_patches_list))])
  # single-turn conversation
  question1 = "Describe this video in detail."
  question = video_prefix + question1
  output1, chat_history = model.chat(tokenizer, pixel_values, question, generation_config, num_patches_list=num_patches_list, history=None, return_history=True)
  print(output1)
  
  # multi-turn conversation
  question2 = "How many people appear in the video?"
  output2, chat_history = model.chat(tokenizer, pixel_values, question, generation_config, num_patches_list=num_patches_list, history=chat_history, return_history=True)
  
  print(output2)
```

## ✏️ Citation

```bibtex

@article{wang2025internvideo,
  title={InternVideo2.5: Empowering Video MLLMs with Long and Rich Context Modeling},
  author={Wang, Yi and Li, Xinhao and Yan, Ziang and He, Yinan and Yu, Jiashuo and Zeng, Xiangyu and Wang, Chenting and Ma, Changlian and Huang, Haian and Gao, Jianfei and Dou, Min and Chen, Kai and Wang, Wenhai and Qiao, Yu and Wang, Yali and Wang, Limin},
  journal={arXiv preprint arXiv:2501.12386},
  year={2025}
}
```