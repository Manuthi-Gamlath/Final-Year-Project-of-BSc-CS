# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange
from timm.models.layers import DropPath
from torch import nn
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (BaseModelOutput,
                                           BaseModelOutputWithPooling)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from configuration_intern_vit import InternVisionConfig

try:
    from flash_attn.bert_padding import pad_input, unpad_input
    from flash_attn.flash_attn_interface import \
        flash_attn_varlen_qkvpacked_func
    has_flash_attn = True
except:
    print('FlashAttention2 is not installed.')
    has_flash_attn = False

logger = logging.get_logger(__name__)


class FlashAttention(nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """

    def __init__(self, softmax_scale=None, attention_dropout=0.0, device=None, dtype=None):
        super().__init__()
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout

    def forward(self, qkv, key_padding_mask=None, causal=False, cu_seqlens=None,
                max_s=None, need_weights=False):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            qkv: The tensor containing the query, key, and value. (B, S, 3, H, D) if key_padding_mask is None
                if unpadded: (nnz, 3, h, d)
            key_padding_mask: a bool tensor of shape (B, S)
        """
        assert not need_weights
        assert qkv.dtype in [torch.float16, torch.bfloat16]
        assert qkv.is_cuda

        if cu_seqlens is None:
            batch_size = qkv.shape[0]
            seqlen = qkv.shape[1]
            if key_padding_mask is None:
                qkv = rearrange(qkv, 'b s ... -> (b s) ...')
                max_s = seqlen
                cu_seqlens = torch.arange(0, (batch_size + 1) * seqlen, step=seqlen, dtype=torch.int32,
                                          device=qkv.device)
                output = flash_attn_varlen_qkvpacked_func(
                    qkv, cu_seqlens, max_s, self.dropout_p if self.training else 0.0,
                    softmax_scale=self.softmax_scale, causal=causal
                )
                output = rearrange(output, '(b s) ... -> b s ...', b=batch_size)
            else:
                nheads = qkv.shape[-2]
                x = rearrange(qkv, 'b s three h d -> b s (three h d)')
                x_unpad, indices, cu_seqlens, max_s = unpad_input(x, key_padding_mask)
                x_unpad = rearrange(x_unpad, 'nnz (three h d) -> nnz three h d', three=3, h=nheads)
                output_unpad = flash_attn_varlen_qkvpacked_func(
                    x_unpad, cu_seqlens, max_s, self.dropout_p if self.training else 0.0,
                    softmax_scale=self.softmax_scale, causal=causal
                )
                output = rearrange(pad_input(rearrange(output_unpad, 'nnz h d -> nnz (h d)'),
                                             indices, batch_size, seqlen),
                                   'b s (h d) -> b s h d', h=nheads)
        else:
            assert max_s is not None
            output = flash_attn_varlen_qkvpacked_func(
                qkv, cu_seqlens, max_s, self.dropout_p if self.training else 0.0,
                softmax_scale=self.softmax_scale, causal=causal
            )

        return output, None


class InternRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


try:
    from apex.normalization import FusedRMSNorm

    InternRMSNorm = FusedRMSNorm  # noqa

    logger.info('Discovered apex.normalization.FusedRMSNorm - will use it instead of InternRMSNorm')
except ImportError:
    # using the normal InternRMSNorm
    pass
except Exception:
    logger.warning('discovered apex but it failed to load, falling back to InternRMSNorm')
    pass


NORM2FN = {
    'rms_norm': InternRMSNorm,
    'layer_norm': nn.LayerNorm,
}


import torch
import torch.nn as nn
import torch.nn.functional as F

class InternVisionEmbeddings(nn.Module):  
    # This module converts image pixels into patch embeddings for a vision transformer-like model.

    def __init__(self, config: InternVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size  # The size of the embedding vector for each patch.
        self.image_size = config.image_size  # The size of the input image (assumed square).
        self.patch_size = config.patch_size  # The size of each square patch.

        # Learnable class embedding (CLS token), which will be prepended to the patch embeddings.The CLS token acts as a global representation of the entire image.
        self.class_embedding = nn.Parameter(
            torch.randn(1, 1, self.embed_dim),  # Shape: (1, 1, hidden_size)
        )

        # Convolutional layer to extract patch embeddings from the image.
        # It acts as a sliding window that splits the image into non-overlapping patches
        # and projects them into `embed_dim`-dimensional vectors.
        self.patch_embedding = nn.Conv2d(
            in_channels=3,  # Input is an RGB image (3 channels).
            out_channels=self.embed_dim,  # Output embedding dimension.
            kernel_size=self.patch_size,  # Patch size determines the receptive field.
            stride=self.patch_size  # Ensures non-overlapping patches.
        )

        # Compute the number of patches per image: (image_size / patch_size)²
        self.num_patches = (self.image_size // self.patch_size) ** 2  
        
        # Total number of positions, including the class token.
        self.num_positions = self.num_patches + 1  

        # Learnable positional embeddings to provide spatial information.
        self.position_embedding = nn.Parameter(torch.randn(1, self.num_positions, self.embed_dim))

    def _get_pos_embed(self, pos_embed, H, W):
        """
        Adjusts the positional embeddings for different image sizes.
        Uses bicubic interpolation to scale the positional embeddings to match the
        new height (H) and width (W) of the patch grid.
        """
        target_dtype = pos_embed.dtype  # Preserve the original data type.
        
        # Reshape positional embeddings from (1, num_patches+1, embed_dim) 
        # to (1, height, width, embed_dim) format.
        pos_embed = pos_embed.float().reshape(
            1, self.image_size // self.patch_size, self.image_size // self.patch_size, -1
        ).permute(0, 3, 1, 2)  # Convert to (1, embed_dim, height, width) format.

        # Interpolate positional embeddings to match the required (H, W) dimensions.
        pos_embed = F.interpolate(pos_embed, size=(H, W), mode='bicubic', align_corners=False)
        
        # Reshape back to (1, H * W, embed_dim) format.
        pos_embed = pos_embed.reshape(1, -1, H * W).permute(0, 2, 1).to(target_dtype)
        
        return pos_embed

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        """
        Forward pass: Converts an input image tensor into patch embeddings.
        """
        target_dtype = self.patch_embedding.weight.dtype  # Ensure dtype consistency.

        # Apply the patch embedding layer (convolution) to extract patch features.
        patch_embeds = self.patch_embedding(pixel_values)  # Shape: (batch_size, embed_dim, height, width)

        batch_size, _, height, width = patch_embeds.shape  # Extract batch size and spatial dimensions. this is after converting to patches

        # Flatten patches: Change from (batch_size, embed_dim, height, width) 
        # to (batch_size, num_patches, embed_dim).
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        # Expand the learnable class embedding across the batch dimension.
        class_embeds = self.class_embedding.expand(batch_size, 1, -1).to(target_dtype)

        # Concatenate the class embedding with the patch embeddings along the sequence dimension.
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)

        # Compute the position embeddings:
        # - The first position embedding is for the class token.
        # - The rest are interpolated to match the patch grid size.
        position_embedding = torch.cat([
            self.position_embedding[:, :1, :],  # Class token positional embedding.
            self._get_pos_embed(self.position_embedding[:, 1:, :], height, width)  # Patch positional embeddings.
        ], dim=1)

        # Add positional embeddings to the patch embeddings.
        embeddings = embeddings + position_embedding.to(target_dtype)

        return embeddings  # Final shape: (batch_size, num_positions, embed_dim)



class InternAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: InternVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.use_flash_attn = config.use_flash_attn and has_flash_attn
        if config.use_flash_attn and not has_flash_attn:
            print('Warning: Flash Attention is not available, use_flash_attn is set to False.')
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f'embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:'
                f' {self.num_heads}).'
            )

        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=config.qkv_bias)
        self.attn_drop = nn.Dropout(config.attention_dropout)
        self.proj_drop = nn.Dropout(config.dropout)

        self.qk_normalization = config.qk_normalization

        if self.qk_normalization:
            self.q_norm = InternRMSNorm(self.embed_dim, eps=config.layer_norm_eps)
            self.k_norm = InternRMSNorm(self.embed_dim, eps=config.layer_norm_eps)

        if self.use_flash_attn:
            self.inner_attn = FlashAttention(attention_dropout=config.attention_dropout)
        self.proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _naive_attn(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        if self.qk_normalization:
            B_, H_, N_, D_ = q.shape
            q = self.q_norm(q.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)
            k = self.k_norm(k.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)

        attn = ((q * self.scale) @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def _flash_attn(self, x, key_padding_mask=None, need_weights=False):
        qkv = self.qkv(x)
        qkv = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, h=self.num_heads)

        if self.qk_normalization:
            q, k, v = qkv.unbind(2)
            q = self.q_norm(q.flatten(-2, -1)).view(q.shape)
            k = self.k_norm(k.flatten(-2, -1)).view(k.shape)
            qkv = torch.stack([q, k, v], dim=2)

        context, _ = self.inner_attn(
            qkv, key_padding_mask=key_padding_mask, need_weights=need_weights, causal=False
        )
        outs = self.proj(rearrange(context, 'b s h d -> b s (h d)'))
        outs = self.proj_drop(outs)
        return outs

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self._naive_attn(hidden_states) if not self.use_flash_attn else self._flash_attn(hidden_states)
        return x


class InternMLP(nn.Module):
    def __init__(self, config: InternVisionConfig):
        super().__init__()
        self.config = config
        self.act = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class InternVisionEncoderLayer(nn.Module):
    def __init__(self, config: InternVisionConfig, drop_path_rate: float):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.norm_type = config.norm_type

        self.attn = InternAttention(config)
        self.mlp = InternMLP(config)
        self.norm1 = NORM2FN[self.norm_type](self.embed_dim, eps=config.layer_norm_eps)
        self.norm2 = NORM2FN[self.norm_type](self.embed_dim, eps=config.layer_norm_eps)

        self.ls1 = nn.Parameter(config.initializer_factor * torch.ones(self.embed_dim))
        self.ls2 = nn.Parameter(config.initializer_factor * torch.ones(self.embed_dim))
        self.drop_path1 = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.drop_path2 = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(
            self,
            hidden_states: torch.Tensor,
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor], Optional[Tuple[torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]`): input to the layer of shape `(batch, seq_len, embed_dim)`
        """
        hidden_states = hidden_states + self.drop_path1(self.attn(self.norm1(hidden_states).to(hidden_states.dtype)) * self.ls1)

        hidden_states = hidden_states + self.drop_path2(self.mlp(self.norm2(hidden_states).to(hidden_states.dtype)) * self.ls2)

        return hidden_states


class InternVisionEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self-attention layers. Each layer is a
    [`InternEncoderLayer`].

    Args:
        config (`InternConfig`):
            The corresponding vision configuration for the `InternEncoder`.
    """

    def __init__(self, config: InternVisionConfig):
        super().__init__()
        self.config = config

        # stochastic depth decay rule: generates a list of drop path rates
        # This list controls how much "depth" is skipped during training to regularize the model
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.num_hidden_layers)]
        
        # Create a list of `InternVisionEncoderLayer` layers, one for each hidden layer
        self.layers = nn.ModuleList([
            InternVisionEncoderLayer(config, dpr[idx]) for idx in range(config.num_hidden_layers)
        ])
        
        # Enabling gradient checkpointing to reduce memory usage during backpropagation
        self.gradient_checkpointing = True

    def forward(
            self,
            inputs_embeds,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Embedded representation of the inputs. Should be float, not int tokens.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        
        # Default value for `output_hidden_states` if not provided
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        
        # Default value for `return_dict` if not provided
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Initialize the encoder states tuple for storing hidden states if `output_hidden_states` is True
        encoder_states = () if output_hidden_states else None
        hidden_states = inputs_embeds  # Start with the input embeddings as the initial hidden states

        # Iterate over all layers of the encoder
        for idx, encoder_layer in enumerate(self.layers):
            
            # If we need to output hidden states, store the current hidden states
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            
            # Apply gradient checkpointing if it's enabled and the model is in training mode
            if self.gradient_checkpointing and self.training:
                # Checkpointing reduces memory by not saving intermediate results of this layer
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    encoder_layer,
                    hidden_states
                )
            else:
                # Normal forward pass through the encoder layer
                layer_outputs = encoder_layer(
                    hidden_states,
                )
            
            # Update the hidden states with the outputs of the current layer
            hidden_states = layer_outputs

        # If we need to return hidden states, add the final layer's hidden states
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        # If we don't want to return a dictionary, return a tuple containing the last hidden states
        # and encoder states (if they exist)
        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states] if v is not None)

        # Otherwise, return a structured output with the last hidden states and encoder states
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states
        )



class InternVisionModel(PreTrainedModel): # this is the top model of the vision part
    main_input_name = 'pixel_values'
    _supports_flash_attn_2 = True
    config_class = InternVisionConfig
    _no_split_modules = ['InternVisionEncoderLayer']

    def __init__(self, config: InternVisionConfig):
        super().__init__(config)
        self.config = config

        self.embeddings = InternVisionEmbeddings(config)
        self.encoder = InternVisionEncoder(config)

    def resize_pos_embeddings(self, old_size, new_size, patch_size):
        pos_emb = self.embeddings.position_embedding
        _, num_positions, embed_dim = pos_emb.shape
        cls_emb = pos_emb[:, :1, :]
        pos_emb = pos_emb[:, 1:, :].reshape(1, old_size // patch_size, old_size // patch_size, -1).permute(0, 3, 1, 2)
        pos_emb = F.interpolate(pos_emb.float(), size=new_size // patch_size, mode='bicubic', align_corners=False)
        pos_emb = pos_emb.to(cls_emb.dtype).reshape(1, embed_dim, -1).permute(0, 2, 1)
        pos_emb = torch.cat([cls_emb, pos_emb], dim=1)
        self.embeddings.position_embedding = nn.Parameter(pos_emb)
        self.embeddings.image_size = new_size
        logger.info('Resized position embeddings from {} to {}'.format(old_size, new_size))

    def get_input_embeddings(self):
        return self.embeddings
        


    def forward(
            self,
            pixel_values: Optional[torch.FloatTensor] = None, #pixel_values: Raw image pixel values as a 4D tensor (batch_size, channels, height, width).
            output_hidden_states: Optional[bool] = None,      #Whether to return hidden states from all layers. Defaults to self.config.output_hidden_states.
            return_dict: Optional[bool] = None,               #Whether to return outputs as a dictionary (BaseModelOutputWithPooling). Defaults to self.config.use_return_dict.
            pixel_embeds: Optional[torch.FloatTensor] = None, #Precomputed image embeddings. If provided, it bypasses pixel_values processing.
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        ) #If output_hidden_states and return_dict are None, they are set to default values from self.config
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None and pixel_embeds is None:
            raise ValueError('You have to specify pixel_values or pixel_embeds') #If both pixel_values and pixel_embeds are None, an error is raised, ensuring at least one is provided.

        if pixel_embeds is not None:
            hidden_states = pixel_embeds #If pixel_embeds is provided, it is used directly as the hidden_states.
        else:
            if len(pixel_values.shape) == 4: #if pixel_values is provided, it is passed through self.embeddings(pixel_values)
                hidden_states = self.embeddings(pixel_values)
            else:
                raise ValueError(f'wrong pixel_values size: {pixel_values.shape}')
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )  #The embeddings (hidden_states) are passed to the transformer encoder. encoder is InternVisionEncoder(config)
        last_hidden_state = encoder_outputs.last_hidden_state #The output embeddings from the encoder.
        pooled_output = last_hidden_state[:, 0, :] #The first token's embedding (typically used as a summary representation).

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
