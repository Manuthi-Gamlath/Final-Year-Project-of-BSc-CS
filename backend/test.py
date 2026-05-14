import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np

from configuration_internvl_chat import InternVLChatConfig
from modeling_internvl_model_language3 import InternVLChatModel
from src.dataset import train_and_valDataset
from src.utils import set_random_seed

def load_pretrained_weights_safetensors(model, safetensors_dir):
    shard_files = sorted(glob.glob(os.path.join(safetensors_dir, "*.pt")))
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
        #print(f" {k}: matched shape {v.shape}")
        #print(model_state_dict.keys())
        if k in model_state_dict:
            expected_shape = model_state_dict[k].shape
            
            if v.shape == expected_shape:
                filtered_state_dict[k] = v
                print(f"✅ {k}: matched shape {v.shape}")
            else:
                print(f"⚠️  {k}: shape mismatch — model: {expected_shape}, checkpoint: {v.shape}")
        #else:
        #    print(f"❌ {k}: not found in model")

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

def load_model_from_pt(model, pt_path):
    print(f"Loading model weights from {pt_path}")
    state_dict = torch.load(pt_path, map_location='cuda')
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded model. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
    return model


def evaluate_model(model, tokenizer, dataloader):
    model.eval()
    predictions = []
    references = []

    IMG_START_TOKEN, IMG_END_TOKEN, IMG_CONTEXT_TOKEN = '<img>', '</img>', '<IMG_CONTEXT>'
    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    MAX_LEN = 1100
    with torch.no_grad():
        for inputs in tqdm(dataloader, desc="Evaluating"):
            expert, novice, label = inputs
            expert = expert.cuda().to(torch.bfloat16)
            novice = novice.cuda().to(torch.bfloat16)
            
            input_modal_video = torch.cat([expert, novice], dim=1) 

            batch_size = expert.shape[0]
            query_template = "".join([
                f"Frame{i+1}: {IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * 16}{IMG_END_TOKEN}\n"
                for i in range(16)
            ])+ "expert and novice are interacting in provided two segment, Eight frames from each segment are sampled from 60 frames, and the task is to predict the engagement values of the expert in the video for all sampled frames, based on social signals of both expert and novice. "
            #print(query_template)
            queries = [query_template for _ in range(batch_size)]
            
            model_inputs = tokenizer(
                queries,
                return_tensors='pt',
                padding="max_length",
                truncation=True,
                max_length=MAX_LEN
            ).to('cuda')
            #generation_config['eos_token_id'] = eos_token_id
            #generation_config = dict(
            #    do_sample=False,
            #    temperature=0.0,
            #    max_new_tokens=756,
            #    top_p=0.1,
            #    num_beams=1
            #)
            eos_token_id=2
            generated_ids = model.generate(
                pixel_values=input_modal_video,
                input_ids=model_inputs["input_ids"],
                attention_mask=model_inputs["attention_mask"],
                img_context_token_id=img_context_token_id,
                max_new_tokens=MAX_LEN,
                top_p=0.1,
                temperature=0.0,
                num_beams=1,
                eos_token_id=eos_token_id,
                do_sample=False
            )

            decoded_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            predictions.extend(decoded_preds)
            references.extend(label)
            #print(decoded_preds)
            break

    return predictions, references

model_path = 'OpenGVLab/InternVideo2_5_Chat_8B'
if __name__ == "__main__":
    torch.manual_seed(42)
    set_random_seed(42)

    # === Modify the saved dir here ===
    save_dir = "InternVideo2.5"
    pt_model_path = f"./output_model/{save_dir}/model_epoch.pt"

    # Tokenizer
    #special_tokens = {"additional_special_tokens": ['<img>', '</img>', '<IMG_CONTEXT>']}
    float_tokens = [f"{x:.2f}" for x in np.arange(0.0, 1.01, 0.01)]
    special_tokens = {
        "additional_special_tokens": ['<img>', '</img>', '<IMG_CONTEXT>'] + float_tokens
    }
    tokenizer = AutoTokenizer.from_pretrained("OpenGVLab/InternVideo2_5_Chat_8B", trust_remote_code=True)
    tokenizer.add_special_tokens(special_tokens)

    # Model
    #config = InternVLChatConfig()
    #model = InternVLChatModel(config).cuda().to(torch.bfloat16)
    #model = InternVLChatModel.from_pretrained(model_path, trust_remote_code=True).cuda().to(torch.bfloat16)
    
    config = InternVLChatConfig.from_pretrained("my_saved_config")
    model = InternVLChatModel(config).cuda().to(torch.bfloat16)
    model = load_model_from_pt(model, pt_model_path)
    
    #config = model.config
    #config.save_pretrained("my_saved_config")
    #print(f"Model loaded from {config}")
    
   
    # Dataset (change to test set paths)
    modalities = [".audio.soundnet.stream"]
    annotation_path = "/mnt/dhanujaw/Noxi/test/"
    data_path = "/mnt/dhanujaw/Noxi/stream_videos_8fps/test/"
    gt_path = "/mnt/dhanujaw/Noxi/GT/test/"
    annotation_path = "/mnt/dhanujaw/Noxi/train/"
    data_path = '/mnt/dhanujaw/Noxi/stream_videos_8fps/train/'
    gt_path = "/mnt/dhanujaw/Noxi/GT/train/"
    test_id_list = ['025', '002', '005', '034', '063', '006', '014', '016']  # Adjust if needed

    test_dataset = train_and_valDataset(annotation_path, data_path, gt_path, test_id_list, modalities)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)

    # Run Evaluation
    preds, refs = evaluate_model(model, tokenizer, test_loader)

    # Print results
    for i in range(min(10, len(preds))):
        print(f"[GT] {refs[i]}")
        print(f"[PR] {preds[i]}")
        print("---")

    # Optionally: save to file
    with open("predictions.txt", "w") as f:
        for pred, ref in zip(preds, refs):
            f.write(f"[GT] {ref}\n[PR] {pred}\n---\n")
