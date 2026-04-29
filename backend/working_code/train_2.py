import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import logging
import argparse
import numpy as np
import os, sys, glob
from transformers import AutoTokenizer
from safetensors.torch import load_file

from configuration_internvl_chat import InternVLChatConfig
from modeling_internvl_model_language import InternVLChatModel
from src.utils import set_random_seed
from src.dataset import train_and_valDataset

# Arg parser
def parse_arguments():
    parser = argparse.ArgumentParser(description='Prefix-Only Training')
    parser.add_argument('--save_dir', type=str, default='PrefixOnly_CEAM')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seed_value', type=int, default=42)
    return parser.parse_args()

# Load weights (safetensors)
def load_pretrained_weights_safetensors(model, safetensors_dir):
    shard_files = sorted(glob.glob(os.path.join(safetensors_dir, "*.safetensors")))
    full_state_dict = {}
    for shard_file in shard_files:
        print(f"Loading {shard_file}")
        shard_state = load_file(shard_file)
        full_state_dict.update(shard_state)

    model_state_dict = model.state_dict()
    filtered_state_dict = {k: v for k, v in full_state_dict.items()
                           if k in model_state_dict and model_state_dict[k].shape == v.shape}

    missing, unexpected = model.load_state_dict(filtered_state_dict, strict=False)
    print(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

# Main training loop
if __name__ == "__main__":
    args = parse_arguments()
    set_random_seed(args.seed_value)

    os.makedirs(f'./output_model/{args.save_dir}', exist_ok=True)
    logging.basicConfig(filename=f'./output_model/{args.save_dir}/train.log', level=logging.INFO)

    tokenizer = AutoTokenizer.from_pretrained('OpenGVLab/InternVideo2_5_Chat_8B', trust_remote_code=True)

    # Dataset paths
    modalities = [".audio.soundnet.stream"]
    annotation_path = "/mnt/dhanujaw/Noxi/train/"
    data_path = '/mnt/dhanujaw/Noxi/stream_videos/train/'
    gt_path = "/mnt/dhanujaw/Noxi/GT/train/"

    train_id_list = ['003', '028', '020', '040', '050', '030', '042', '023', '046',
                     '013', '064', '026', '056', '068', '001', '055', '045', '072', '052',
                     '077', '021', '047', '044', '049', '029', '070', '069', '038', '022',
                     '048', '067', '057', '027', '043', '039', '073', '066']

    train_dataset = train_and_valDataset(annotation_path, data_path, gt_path, train_id_list, modalities)
    trainloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1, pin_memory=True)

    config = InternVLChatConfig()
    model = InternVLChatModel(config).half().cuda()

    load_pretrained_weights_safetensors(model, "./")

    optimizer = optim.AdamW(model.parameters(), lr=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    scaler = GradScaler()

    IMG_START_TOKEN, IMG_END_TOKEN, IMG_CONTEXT_TOKEN = '<img>', '</img>', '<IMG_CONTEXT>'

    for epoch in range(args.epochs):
        model.train()
        loss_meter = []

        pbar = tqdm(trainloader, desc=f"Epoch [{epoch+1}/{args.epochs}]", leave=True)

        for inputs in pbar:
            expert, novice, label = inputs
            expert = expert.half().cuda()

            query = "".join([f"Frame{i+1}: {IMG_START_TOKEN}{IMG_CONTEXT_TOKEN*16}{IMG_END_TOKEN}\n" for i in range(32)])

            model_inputs = tokenizer(query, return_tensors='pt', padding="max_length", truncation=True, max_length=512).to('cuda')

            input_ids = model_inputs['input_ids']
            attention_mask = model_inputs['attention_mask']

            labels = input_ids.clone()
            labels[:, :-1] = input_ids[:, 1:]
            labels[:, -1] = -100

            img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)

            optimizer.zero_grad()

            with autocast(dtype=torch.float16):
                output = model(pixel_values=expert, input_ids=input_ids, attention_mask=attention_mask,
                               img_context_token_id=img_context_token_id, labels=labels)
                loss = output.loss

            scaler.scale(loss).backward()
            #scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            #scaler.step(optimizer)
            #scaler.update()

            loss_meter.append(loss.item())
            pbar.set_postfix({'loss': loss.item()})

        epoch_loss = np.mean(loss_meter)
        scheduler.step(epoch_loss)
        logging.info(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")

        torch.save(model.state_dict(), f'./output_model/{args.save_dir}/model_epoch_{epoch+1}.pt')

    print("Training completed.")