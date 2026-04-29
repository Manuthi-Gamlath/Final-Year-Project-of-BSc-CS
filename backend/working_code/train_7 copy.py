# this is similar to exact datas processing from internvideo2.5
import os, sys, glob
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.amp import autocast
from tqdm import tqdm
import logging
import argparse
import numpy as np
from transformers import AutoTokenizer
from safetensors.torch import load_file
import transformers
from configuration_internvl_chat import InternVLChatConfig
from modeling_internvl_model_language3 import InternVLChatModel
from src.utils import set_random_seed
from src.dataset import train_and_valDataset
from conversation import get_conv_template
from constants import (CLIP_MEAN, CLIP_STD, IMAGENET_MEAN, IMAGENET_STD,
                        IMG_CONTEXT_TOKEN, IMG_END_TOKEN, IMG_START_TOKEN,
                        SIGLIP_MEAN, SIGLIP_STD)
from transformers.trainer_pt_utils import LabelSmoother
torch.set_printoptions(threshold=float('inf'))

IGNORE_TOKEN_ID = LabelSmoother.ignore_index
model_path = 'OpenGVLab/InternVideo2_5_Chat_8B'

# Arg parser
def parse_arguments():
    parser = argparse.ArgumentParser(description='Prefix-Only Training')
    parser.add_argument('--save_dir', type=str, default='InternVideo2.5')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seed_value', type=int, default=42)
    return parser.parse_args()

def load_pretrained_weights_safetensors(model, safetensors_dir):
    shard_files = sorted(glob.glob(os.path.join(safetensors_dir, "*.safetensors")))
    full_state_dict = {}
    
    for shard_file in shard_files:
        print(f"Loading {shard_file}")
        shard_state = load_file(shard_file)
        full_state_dict.update(shard_state)

    model_state_dict = model.state_dict()
    print(f"\nTotal keys in model: {len(model_state_dict)}")
    print(f"Total keys in loaded state dict: {len(full_state_dict)}")

    # Filter and log shape comparisons
    filtered_state_dict = {}
    print("\n--- Matching keys with same shape ---")
    for k, v in full_state_dict.items():
        if k in model_state_dict:
            expected_shape = model_state_dict[k].shape
            
            if v.shape == expected_shape:
                filtered_state_dict[k] = v
                print(f"✅ {k}: matched shape {v.shape}")
            else:
                print(f"⚠️  {k}: shape mismatch — model: {expected_shape}, checkpoint: {v.shape}")

    # Load filtered weights
    missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
    matched_keys = list(filtered_state_dict.keys())

    # Split by module type
    matched_vision = [k for k in matched_keys if k.startswith("vision_model.")]
    matched_language = [k for k in matched_keys if k.startswith("language_model.")]
    missing_vision = [k for k in missing_keys if k.startswith("vision_model.")]
    missing_language = [k for k in missing_keys if k.startswith("language_model.")]

    # Summary
    print(f"\n✅ Matched keys: {len(matched_keys)}")
    print(f"❌ Missing keys: {len(missing_keys)}")
    print(f"❌ Unexpected keys: {len(unexpected_keys)}")

    print(f"\n🔍 Matched keys in vision part: {len(matched_vision)}")
    print(f"🔍 Matched keys in language part: {len(matched_language)}")
    print(f"🔍 Missing keys in vision part: {len(missing_vision)}")
    print(f"🔍 Missing keys in language part: {len(missing_language)}")


def extract_reason_and_prediction(label_texts):
    result = []
    for full in label_texts:
        # Example rule-based split (customize as needed)
        if "Frame-level engagement:" in full:
            parts = full.split("Frame-level engagement:")
            reasoning = parts[0].strip()
            prediction = "Frame-level engagement:" + parts[1].strip()
        else:
            reasoning = full.strip()
            prediction = ""
        result.append((reasoning, prediction))
    return result


def preprocess_internlm(
    template_name,
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    num_image_token_list: list,
    text_only: bool = False,
    group_by_length: bool = False,
    use_packed_ds: bool = False,
    ds_name: str = None,
    num_image: int = 1,
    model_max_length=None
):
    if model_max_length is None:
        model_max_length = tokenizer.model_max_length
    #ssssssssss",model_max_length)
    conv = get_conv_template(template_name)
    roles = {'human': conv.roles[0], 'gpt': conv.roles[1]}
    conversations = []

    for i, source in enumerate(sources):
        if roles[source[0]['from']] != conv.roles[0]:
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence['from']]
            assert role == conv.roles[j % 2], f'{i}'
            sentence['value'] = sentence['value'].strip()
            conv.append_message(role, sentence['value'])
        conversations.append(conv.get_prompt())

    if not text_only:
        new_conversations = []
        for conversation in conversations:
            for i in range(num_image):
                image_tokens = f'{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * num_image_token_list[i]}{IMG_END_TOKEN}'
                conversation = conversation.replace('<image>', image_tokens, 1)
            new_conversations.append(conversation)
        conversations = new_conversations
    #print(conversations)
    input_ids = tokenizer(
        conversations,
        return_tensors='pt',
        padding=False if group_by_length or use_packed_ds else 'max_length',
        max_length=model_max_length,
        truncation=True,
    ).input_ids

    targets = input_ids.clone()

    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        cur_len = 1
        target[:cur_len] = IGNORE_TOKEN_ID

        parts = conversation.split(conv.roles[1])
        info = parts[0] + conv.roles[1]
        temp_len = len(tokenizer(info).input_ids) - 1
        target[cur_len: cur_len + temp_len] = IGNORE_TOKEN_ID
        cur_len += temp_len

        for index in range(1, len(parts) - 1):
            info = parts[index]
            part1, part2 = info.split(conv.roles[0])
            temp_len = len(tokenizer(part1).input_ids) - 1
            cur_len += temp_len
            part = conv.roles[0] + part2 + conv.roles[1]
            temp_len = len(tokenizer(part).input_ids) - 1
            target[cur_len: cur_len + temp_len] = IGNORE_TOKEN_ID
            cur_len += temp_len

        last_info = parts[-1]
        temp_len = len(tokenizer(last_info).input_ids) - 1
        cur_len += temp_len

        target[cur_len:] = IGNORE_TOKEN_ID

        # ✅ Unmask <eos> token (last non-padding token) for prediction
        last_idx = (target != tokenizer.pad_token_id).nonzero()[-1].item()
        if target[last_idx] == tokenizer.eos_token_id:
            target[last_idx] = tokenizer.eos_token_id  # allow model to predict it

        if cur_len < model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_TOKEN_ID
                print(f'WARNING: tokenization mismatch: {cur_len} vs. {total_len}. This dataset is {ds_name}.')
                sys.stdout.flush()

    return {
        'input_ids': input_ids,
        'labels': targets,
        'attention_mask': input_ids.ne(tokenizer.pad_token_id),
    }
# Main training loop
if __name__ == "__main__":
    args = parse_arguments()
    set_random_seed(args.seed_value)

    os.makedirs(f'./output_model/{args.save_dir}', exist_ok=True)
    logging.basicConfig(filename=f'./output_model/{args.save_dir}/train.log', level=logging.INFO)

    # Add special tokens
    # Add special tokens
    #float_tokens = [f"{x:.2f}" for x in np.arange(0.0, 1.01, 0.01)]
    special_tokens = {
        "additional_special_tokens": [
            "<img>", "</img>", "<IMG_CONTEXT>"
        ] 
    }
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.add_special_tokens(special_tokens)
    #tokenizer.add_special_tokens(special_tokens)
    

    #tokenizer = AutoTokenizer.from_pretrained('OpenGVLab/InternVideo2_5_Chat_8B', trust_remote_code=True)
    #tokenizer.add_special_tokens(special_tokens)

    # Build model
    config = InternVLChatConfig.from_pretrained("my_saved_config")
    #config.vocab_size = len(tokenizer)
    model = InternVLChatModel(config).half().cuda().to(torch.bfloat16)
    model.language_model.resize_token_embeddings(len(tokenizer))
    print("tokenizer",len(tokenizer))
    #model.resize_token_embeddings(len(tokenizer))
    #assert model.get_input_embeddings().num_embeddings == len(tokenizer), \
    #"❌ Tokenizer and model vocab size mismatch!"

    #float_tokens = [f"{x:.2f}" for x in np.arange(0.0, 1.01, 0.01)]
    #vocab = tokenizer.get_vocab()
    # Write to file
    #with open("tokenizer_vocab.txt", "w") as f:   
    #    f.write("\n### Full Tokenizer Vocabulary ###\n")
    #    for token in vocab:
    #        f.write(f"{token}\n")
    # Find tokens that were not added
    #missing_tokens = [tok for tok in float_tokens if tok not in vocab]
   # 
   # if missing_tokens:
   #     print("❌ Missing tokens:", missing_tokens)
   # else:
   #     print("✅ All float tokens were successfully added!")


    # Dataset paths
    modalities = [".audio.soundnet.stream"]
    annotation_path = "/mnt/dhanujaw/Noxi/train/"
    data_path = '/mnt/dhanujaw/Noxi/stream_videos_8fps/train/'
    gt_path = "/mnt/dhanujaw/Noxi/GT/train/"

    train_id_list = ['003', '028', '020', '040', '050', '030', '042', '023', '046',
                     '013', '064', '026', '056', '068', '001', '055', '045', '072', '052',
                     '077', '021', '047', '044', '049', '029', '070', '069', '038', '022',
                     '048', '067', '057', '027', '043', '039', '073', '066']

    train_dataset = train_and_valDataset(annotation_path, data_path, gt_path, train_id_list, modalities)
    trainloader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0, pin_memory=False)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    IMG_START_TOKEN, IMG_END_TOKEN, IMG_CONTEXT_TOKEN = '<img>', '</img>', '<IMG_CONTEXT>'
    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    query_template_test = "'<|im_start|>system\n你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。<|im_end|>\n<|im_start|>user\n"+"".join([
                f"Frame{i+1}: {IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * 16}{IMG_END_TOKEN}\n" 
                for i in range(16)
            ]) + "expert and novice are interacting in provided two segments, Eight frames from each segment are sampled from 61 frames, and the task is to predict the engagement values of the expert in the video for all sampled frames, based on social signals of both expert and novice. "

    for epoch in range(args.epochs):
        model.train()
        loss_meter = []

        pbar = tqdm(trainloader, desc=f"Epoch [{epoch+1}/{args.epochs}]", leave=True)

        for inputs in pbar:
            expert, novice, label = inputs
            expert = expert.cuda().to(torch.bfloat16)
            novice = novice.cuda().to(torch.bfloat16)
            
            input_modal_video = torch.cat([expert, novice], dim=1) 
            batch_size = expert.shape[0]

            # Generate query string for each sample in the batch
            query_template ="expert and novice are interacting in provided two video segments concataneated as one video, Eight frames from each segment are sampled from 61 frames, and the task is to predict the engagement values of the expert in the video for all sampled frames, based on social signals of both expert and novice. "
            query_template ="expert and novice are interacting in provided two video segments concataneated as one video, Eight frames from each segment are sampled from 61 frames, and the task is to predict the engagement values of the expert in the video for all sampled frames, based on social signals of both expert and novice. "
            new_list=[]
            for i in label:
                data_item ={"id": "video_3827", "video": "video_3827.mp4", "conversations": [{"from": "human", "value": "<image>"+query_template+"\n "}, {"from": "gpt", "value": i}], "duration": 23.77147423092289}
                num_frames=8
                num_video_token = 16
                num_patches = 16
                num_image_tokens = [num_video_token] * num_patches
                image_list=["a"]*16  # Placeholder for image paths
                duration = data_item['duration']
                msg = f""
                special_tokens = msg.strip() + '\n' + '\n'.join(['Frame{}: <image>'.format(i + 1) for i in range(len(image_list))])
                
                if '<image>\n' in data_item['conversations'][0]['value']:
                    data_item['conversations'][0]['value'] = data_item['conversations'][0]['value'].replace(
                        '<image>\n', special_tokens)
                else:
                    data_item['conversations'][0]['value'] = data_item['conversations'][0]['value'].replace(
                        '<image>', special_tokens)
                new_list.append(data_item['conversations'])
           
           # print(query_template)
           # queries = [query_template for _ in range(batch_size)]
            MAX_LEN = 1200
            ret = preprocess_internlm('internvl2_5', new_list,
                                      tokenizer, num_image_tokens, group_by_length=True,
                                      ds_name="", num_image=num_patches, model_max_length=MAX_LEN)
            input_ids=ret['input_ids'].to('cuda')
            labels=ret['labels'].to('cuda')
            attention_mask = ret['attention_mask'].to('cuda')
            #print(input_ids)
            #print(labels)
            #print()
            #print(attention_mask)
            #print(input_ids.shape)
            #print(labels.shape)
            #label_texts = list(label)
            #queries = [query_template+label + tokenizer.eos_token for label in label_texts]
            #print(queries)
            #label_texts = [
            #                    f"<reason>{reasoning_part}</reason><prediction>{prediction_part}</prediction>{tokenizer.eos_token}"
            #                    for (reasoning_part, prediction_part) in extract_reason_and_prediction(label_texts)
            #                ]

            #print('rrrrrrrrrrrrrrrrrrrrrrrrrrrrrr',tokenizer.eos_token,tokenizer.eos_token_id)
            #label_texts = [label + tokenizer.eos_token for label in label_texts]
            
            # Tokenize input queries
            

            ## Tokenize target responses
            test_inputs = tokenizer(
                query_template_test,
                return_tensors='pt',
                padding=False,
                truncation=True,
                max_length=MAX_LEN
            ).to('cuda')
            #print(test_inputs)
            #attention_label[i, query_length:query_length + max_copy_len] = gt_tokens[i,query_length:]
            #print(labels)
            # Debug information
            #print(f"Query length: {query_length}")
            #print(f"Labels shape: {labels.shape}")
            
            # Verify our labels are set correctly
            sample_idx = 0  # Check first sample in batch
            #print(labels)
            #print("👉 Decoded text:", tokenizer.decode(labels[0][400:], skip_special_tokens=False))
            optimizer.zero_grad()
            
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                output = model(
                    pixel_values=input_modal_video, 
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    img_context_token_id=img_context_token_id, 
                    labels=labels,
                    tokenizer=tokenizer
                )
                loss = output.loss

            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            loss_meter.append(loss.item())
            pbar.set_postfix({'loss': loss.item()})
        eos_token_id = tokenizer.eos_token_id
        epoch_loss = np.mean(loss_meter)
        #print(test_inputs["input_ids"])
        generated_ids = model.generate(
                pixel_values=input_modal_video,
                input_ids=test_inputs["input_ids"],
                attention_mask=test_inputs["attention_mask"],
                img_context_token_id=img_context_token_id,
                max_new_tokens=MAX_LEN,
                top_p=0.1,
                temperature=0.0,
                num_beams=1,
                eos_token_id=eos_token_id,
                do_sample=False
            )

        decoded_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
        print(decoded_preds)
        with open("./generated_predictions1.txt", "a", encoding="utf-8") as f:
            for pred in decoded_preds:
                f.write(pred + "\n")
        #print(tokenizer.batch_decode([364, 52599,
        #   410, 19979,  2914,   446,   410,  6358,   435,  2930, 14226,   328,
        #  3285,   519,  3751, 54850,   454,  4191, 11745, 24084,   281,   262], skip_special_tokens=True))
        scheduler.step(epoch_loss)
        current_lr = optimizer.param_groups[0]['lr']
        logging.info(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")
        print(f"🔁 Epoch {epoch+1} | 🔧 Current LR: {current_lr:.6f}")

        torch.save(model.state_dict(), f'./output_model/{args.save_dir}/model_epoch.pt')

    print("Training completed.")