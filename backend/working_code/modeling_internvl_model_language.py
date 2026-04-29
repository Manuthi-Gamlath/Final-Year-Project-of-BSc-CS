import math
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import logging
from configuration_internvl_chat import InternVLChatConfig
from modeling_intern_vit import has_flash_attn
from modeling_internlm2 import InternLM2ForCausalLM

logger = logging.get_logger(__name__)

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

        self.num_image_token = self.num_tome_tokens // self.local_num_frames
        logger.info(f'num_image_token: {self.num_image_token}')

        # ===== Load Language Model =====
        self.language_model = InternLM2ForCausalLM(config.llm_config)

        # ===== Vision-Language Mapping =====
        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.llm_config.hidden_size

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
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vit_embeds = pixel_values
        input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()
        self.img_context_token_id = img_context_token_id

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.view(B * N, C)
        input_ids = input_ids.view(B * N)
        selected = (input_ids == self.img_context_token_id)

        try:
            input_embeds[selected] = vit_embeds.view(-1, C).to(input_embeds.dtype)
        except Exception:
            vit_embeds = vit_embeds.view(-1, C).to(input_embeds.dtype)
            n_token = selected.sum()
            input_embeds[selected] = vit_embeds[:n_token]

        input_embeds = input_embeds.view(B, N, C)

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
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

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
