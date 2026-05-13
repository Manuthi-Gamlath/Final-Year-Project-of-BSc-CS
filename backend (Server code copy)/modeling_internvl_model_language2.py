import math
import os, sys, glob
import torch
from torch import nn
from safetensors.torch import load_file
from torch.nn import CrossEntropyLoss
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import logging
from configuration_internvl_chat import InternVLChatConfig
from modeling_intern_vit import has_flash_attn
from modeling_internlm2 import InternLM2ForCausalLM
from peft import get_peft_model, LoraConfig, TaskType
import numpy as np
logger = logging.get_logger(__name__)

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
    print(full_state_dict.keys())
    # Filter and rename matching keys
    filtered_state_dict = {}
    print("\n--- Matching keys with same shape ---")
    for k, v in full_state_dict.items():
        if k.startswith("language_model.model."):
            k_new = k.replace("language_model.model.", "model.")
        elif k.startswith("language_model.output."):
            k_new = k.replace("language_model.output.", "output.")
        else:
            k_new = k
        if k_new in model_state_dict:
            expected_shape = model_state_dict[k_new].shape
            if v.shape == expected_shape:
                filtered_state_dict[k_new] = v  # ✅ use new key
                print(f"✅ {k_new}: matched shape {v.shape}")
            else:
                print(f"⚠️  {k_new}: shape mismatch — model: {expected_shape}, checkpoint: {v.shape}")
        #else:
            #print(f"❌ {k_new}: not found in model")

    # Load into model
    missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
    #print(missing_keys)
    matched_keys = list(filtered_state_dict.keys())

    # Analysis
    matched_vision = [k for k in matched_keys if k.startswith("vision_model.")]
    matched_language = [k for k in matched_keys if k.startswith("model.layers.") or k.startswith("model.tok_embeddings")]
    missing_vision = [k for k in missing_keys if k.startswith("vision_model.")]
    missing_language = [k for k in missing_keys if k.startswith("model.layers.") or k.startswith("model.tok_embeddings")]

    print(f"\n✅ Matched keys: {len(matched_keys)}")
    print(f"❌ Missing keys: {len(missing_keys)}")
    print(f"❌ Unexpected keys: {len(unexpected_keys)}")

    print(f"\n🔍 Matched keys in vision part: {len(matched_vision)}")
    print(f"🔍 Matched keys in language part: {len(matched_language)}")
    print(f"🔍 Missing keys in vision part: {len(missing_vision)}")
    print(f"🔍 Missing keys in language part: {len(missing_language)}")


import math

def safe_init_lora_weights(model):
    for name, module in model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            for key, lora_param in module.lora_A.items():
                if isinstance(lora_param, torch.nn.Parameter):
                    if torch.isnan(lora_param).any():
                        print(f"⚠️ [NaN detected before] in {name}.lora_A.{key}")
                    try:
                        nn.init.kaiming_uniform_(lora_param.data, a=math.sqrt(5))
                    except Exception as e:
                        print(f"❌ Failed to init {name}.lora_A.{key}: {e}")

            for key, lora_param in module.lora_B.items():
                if isinstance(lora_param, torch.nn.Parameter):
                    if torch.isnan(lora_param).any():
                        print(f"⚠️ [NaN detected before] in {name}.lora_B.{key}")
                    try:
                        nn.init.zeros_(lora_param.data)
                    except Exception as e:
                        print(f"❌ Failed to init {name}.lora_B.{key}: {e}")

class InternVLChatModel(PreTrainedModel):
    config_class = InternVLChatConfig
    main_input_name = 'pixel_values'
    base_model_prefix = 'language_model'

    def __init__(self, config: InternVLChatConfig, use_flash_attn=True):
        super().__init__(config)

        self.local_num_frames = 4
        self.num_tome_tokens = 64
        self.config = config
        self.patch_size = config.vision_config.patch_size
        self.select_layer = config.select_layer
        self.template = config.template
        self.downsample_ratio = config.downsample_ratio

        use_flash_attn = use_flash_attn if has_flash_attn else False
        config.vision_config.use_flash_attn = use_flash_attn
        config.llm_config.attn_implementation = 'flash_attention_2' if use_flash_attn else 'eager'
        config.llm_config.vocab_size = 92665 
        self.num_image_token = self.num_tome_tokens // self.local_num_frames
        logger.info(f'num_image_token: {self.num_image_token}')
        #eos_token_id=92542

        # ===== Load Language Model =====
        self.language_model = InternLM2ForCausalLM(config.llm_config)
        load_pretrained_weights_safetensors(self.language_model, "./")
        self.language_model = self.language_model.half()
        # ===== Dynamically find LoRA target modules =====
        attention_proj_modules = set()
        for name, module in self.language_model.named_modules():
            #print(name)
            if any(proj in name.lower() for proj in ["q", "k", "v", "o"]) and isinstance(module, nn.Linear):
                attention_proj_modules.add(name.split(".")[-1])

        print(f"[LoRA] Using target modules: ",["wo", "wqkv", "w1", "w2"])

        # ===== Apply LoRA =====
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules= ["wo", "wqkv", "w1", "w2"],#attention_proj_modules,#"wo", "w1", "w2", "w3",
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        self.language_model = get_peft_model(self.language_model, lora_config)
        print("🔍 Checking for NaNs right after applying LoRA...")
        for name, param in self.language_model.named_parameters():
            if torch.isnan(param).any():
                print(f"❌ NaN found immediately after LoRA wrapping: {name}")

        ## ===== Reinitialize LoRA weights =====
        safe_init_lora_weights(self.language_model)
        print("inttttttttttttttttttttttt")
        for name, param in self.language_model.named_parameters():
            if torch.any(torch.isnan(param)):
                print(f"❌ NaN found in parameter: {name}")

        ## ===== Freeze LLM except LoRA layers =====
        for name, param in self.language_model.named_parameters():
            #print(name)
            if "lora_" not in name:
            #if "output" not in name:
                param.requires_grad = False
        # Unfreeze LoRA and output layer
        for name, param in self.language_model.named_parameters():
            if  "output.weight" in name:
                param.requires_grad = True
            if "tok_embeddings" in name:
                 param.requires_grad = True
        for name, param in self.language_model.named_parameters():
            if "output.weight" in name:
              print(name, "trainable:", param.requires_grad)
            if "tok_embeddings" in name:
                 print(name, "trainable:", param.requires_grad)
            if "lora_" in name:
              print(name, "trainable:", param.requires_grad)
        self.language_model.print_trainable_parameters()
        # ===== Vision-Language Mapping (if needed elsewhere) =====
        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.llm_config.hidden_size
        print("vit_hidden_size",vit_hidden_size)
        print("llm_hidden_size",config.llm_config.hidden_size)

        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
            nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size)
        )

        self.img_context_token_id = None

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor = None,
        position_ids: torch.LongTensor = None,
        past_key_values=None,
        labels: torch.LongTensor = None,
        use_cache: bool = None,
        output_attentions: bool = None,
        output_hidden_states: bool = None,
        return_dict: bool = None,
        img_context_token_id: int = None,
        tokenizer = None
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Use pixel values directly as image features
        vit_embeds = pixel_values  # shape: (B, N, C)
        input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()
      
        self.img_context_token_id = img_context_token_id

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.view(B * N, C)
        input_ids = input_ids.view(B * N)
        selected = (input_ids == self.img_context_token_id)

        try:
            assert selected.sum() == vit_embeds.numel() // C, f"Mismatch: expected {selected.sum()} video tokens but got {vit_embeds.shape}"
            input_embeds[selected] = vit_embeds.view(-1, C).to(input_embeds.dtype)
        except Exception:
            vit_embeds = vit_embeds.view(-1, C).to(input_embeds.dtype)
            n_token = selected.sum()
            input_embeds[selected] = vit_embeds[:n_token]

        input_embeds = input_embeds.view(B, N, C)
        #print("yyyyyyyyyyyy",input_embeds)
        #if torch.any(input_embeds != 0):
        #    print("✅ input_embeds is non-zero")
        #else:
        #    print("❌ input_embeds is all zeros")
        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        logits = outputs.logits
        ignore_index = -100
        loss = None
        #print(labels)
        if labels is not None:
            if labels.shape != logits.shape[:2]:
                labels = torch.nn.functional.pad(labels, (0, logits.shape[1] - labels.shape[1]), value=ignore_index)
        
            loss_fct = CrossEntropyLoss(ignore_index=ignore_index, reduction='none')
            flat_logits = logits.view(-1, logits.size(-1))
            flat_labels = labels.view(-1)
        
            # Get token ids for reasoning and prediction spans
            reason_start_id = tokenizer.convert_tokens_to_ids("<reason>")
            reason_end_id   = tokenizer.convert_tokens_to_ids("</reason>")
            pred_start_id   = tokenizer.convert_tokens_to_ids("<prediction>")
            pred_end_id     = tokenizer.convert_tokens_to_ids("</prediction>")
        
            batch_size, seq_len = labels.size()
            reason_mask = torch.zeros_like(labels, dtype=torch.bool)
            value_mask  = torch.zeros_like(labels, dtype=torch.bool)
        
            for i in range(batch_size):
                row = labels[i]
        
                rs = (row == reason_start_id).nonzero(as_tuple=True)[0]
                re = (row == reason_end_id).nonzero(as_tuple=True)[0]
                ps = (row == pred_start_id).nonzero(as_tuple=True)[0]
                pe = (row == pred_end_id).nonzero(as_tuple=True)[0]
        
                if len(rs) > 0 and len(re) > 0:
                    reason_mask[i, rs[0]+1 : re[0]] = True
                if len(ps) > 0 and len(pe) > 0:
                    value_mask[i, ps[0]+1 : pe[0]] = True
        
            reason_mask = reason_mask.view(-1)
            value_mask = value_mask.view(-1)
        
            per_token_loss = loss_fct(flat_logits, flat_labels)
        
            reason_loss = per_token_loss[reason_mask].mean() if reason_mask.any() else torch.tensor(0.0, device=logits.device)
            value_loss  = per_token_loss[value_mask].mean()  if value_mask.any()  else torch.tensor(0.0, device=logits.device)
        
            loss = reason_loss + 2.0 * value_loss





        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor = None,
        position_ids: torch.LongTensor = None,
        img_context_token_id: int = None,
        **generate_kwargs
    ):
        self.eval()
        # Get input embeddings
        input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()
        self.img_context_token_id = img_context_token_id

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.view(B * N, C)
        input_ids = input_ids.view(B * N)
        selected = (input_ids == self.img_context_token_id)

        try:
            input_embeds[selected] = pixel_values.view(-1, C).to(input_embeds.dtype)
        except Exception:
            vit_embeds = pixel_values.view(-1, C).to(input_embeds.dtype)
            n_token = selected.sum()
            input_embeds[selected] = vit_embeds[:n_token]

        input_embeds = input_embeds.view(B, N, C)

        # Call LLM generate
        return self.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **generate_kwargs
        )

