from __future__ import annotations
from typing import Tuple
from huggingface_hub import PyTorchModelHubMixin

import torch
import math
import numpy as np
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Module, ModuleList

from einops import rearrange, pack, unpack
from einops.layers.torch import Rearrange

from x_transformers.attend import Attend
from x_transformers import (
    RMSNorm,
    FeedForward
)

# mlp 
def mlp(dim, dim_hidden, dim_out):
    return nn.Sequential(
        nn.Linear(dim, dim_hidden),
        nn.SiLU(),
        nn.Linear(dim_hidden, dim_out),
    )

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def softclamp(t, value):
    return (t / value).tanh() * value

# rmsnorm

class MultiHeadRMSNorm(Module):
    def __init__(self, dim, heads = 1):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(heads, 1, dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.gamma * self.scale

# attention

class JointAttention(Module):
    def __init__(
        self,
        *,
        dim,
        dim_inputs: Tuple[int, ...],
        dim_head = 64,
        heads = 8,
        qk_rmsnorm = True,
        flash = True,
        softclamp = False,
        softclamp_value = 50.,
        attend_kwargs: dict = dict()
    ):
        super().__init__()
        """
        ein notation

        b - batch
        h - heads
        n - sequence
        d - feature dimension
        """

        dim_inner = dim_head * heads

        num_inputs = len(dim_inputs)
        self.num_inputs = num_inputs

        self.to_qkv = ModuleList([nn.Linear(dim_input, dim_inner * 3, bias = False) for dim_input in dim_inputs])

        self.split_heads = Rearrange('b n (qkv h d) -> qkv b h n d', h = heads, qkv = 3)

        self.attend = Attend(
            flash = flash,
            softclamp_logits = softclamp,
            logit_softclamp_value = softclamp_value,
            **attend_kwargs
        )

        self.merge_heads = Rearrange('b h n d -> b n (h d)')

        self.to_out = ModuleList([nn.Linear(dim_inner, dim_input, bias = False) for dim_input in dim_inputs])

        self.qk_rmsnorm = qk_rmsnorm
        self.q_rmsnorms = (None,) * num_inputs
        self.k_rmsnorms = (None,) * num_inputs

        if qk_rmsnorm:
            self.q_rmsnorms = ModuleList([MultiHeadRMSNorm(dim_head, heads = heads) for _ in range(num_inputs)])
            self.k_rmsnorms = ModuleList([MultiHeadRMSNorm(dim_head, heads = heads) for _ in range(num_inputs)])

        self.register_buffer('dummy', torch.tensor(0), persistent = False)

    def forward(
        self,
        inputs: Tuple[Tensor],
        masks: Tuple[Tensor | None] | None = None
    ):

        device = self.dummy.device

        assert len(inputs) == self.num_inputs

        masks = default(masks, (None,) * self.num_inputs)

        # project each modality separately for qkv
        # also handle masks, assume None means attend to all tokens

        all_qkvs = []
        all_masks = []

        for x, mask, to_qkv, q_rmsnorm, k_rmsnorm in zip(inputs, masks, self.to_qkv, self.q_rmsnorms, self.k_rmsnorms):

            qkv = to_qkv(x)
            qkv = self.split_heads(qkv)

            # optional qk rmsnorm per modality

            if self.qk_rmsnorm:
                q, k, v = qkv
                q = q_rmsnorm(q)
                k = k_rmsnorm(k)
                qkv = torch.stack((q, k, v))

            all_qkvs.append(qkv)

            # handle mask per modality

            if not exists(mask):
                mask = torch.ones(x.shape[:2], device = device, dtype = torch.bool)

            all_masks.append(mask)

        # combine all qkv and masks

        all_qkvs, packed_shape = pack(all_qkvs, 'qkv b h * d')
        all_masks, _ = pack(all_masks, 'b *')

        # attention

        q, k, v = all_qkvs

        outs, *_ = self.attend(q, k, v, mask = all_masks)

        # merge heads and then separate by modality for combine heads projection

        outs = self.merge_heads(outs)
        outs = unpack(outs, packed_shape, 'b * d')

        # separate combination of heads for each modality

        all_outs = []

        for out, to_out in zip(outs, self.to_out):
            out = to_out(out)
            all_outs.append(out)

        return tuple(all_outs)

# adaptive layernorm
# aim for clarity in generalized version

class AdaptiveLayerNorm(Module):
    def __init__(
        self,
        dim,
        dim_cond = None
    ):
        super().__init__()
        has_cond = exists(dim_cond)
        self.has_cond = has_cond

        self.ln = nn.LayerNorm(dim, elementwise_affine = not has_cond)
 
        if has_cond:
            cond_linear = nn.Linear(dim_cond, dim * 2)

            self.to_cond = nn.Sequential(
                Rearrange('b d -> b 1 d'),
                nn.SiLU(),
                cond_linear
            )

            nn.init.zeros_(cond_linear.weight)

            nn.init.constant_(cond_linear.bias[:dim], 1.)
            nn.init.zeros_(cond_linear.bias[dim:])

    def forward(
        self,
        x,
        cond = None
    ):
        assert not (exists(cond) ^ self.has_cond), 'condition must be passed in if dim_cond is set at init. it should not be passed in if not set'

        x = self.ln(x)

        if self.has_cond:
            gamma, beta = self.to_cond(cond).chunk(2, dim = -1)
            x = x * gamma + beta

        return x

# class

class MMDiTBlock(Module):
    def __init__(
        self,
        *,
        dim_joint_attn,
        dim_modalities: Tuple[int, ...],
        dim_conds: Tuple[int, ...],
        dim_head = 64,
        heads = 8,
        qk_rmsnorm = True,
        flash_attn = True,
        softclamp = False,
        softclamp_value = 50.,
        ff_kwargs: dict = dict()
    ):
        super().__init__()
        self.num_modalities = len(dim_modalities)
        self.dim_modalities = dim_modalities

        # handle optional time conditioning

        has_cond_array = [exists(dim_cond) for dim_cond in dim_conds]
        self.has_cond_array = has_cond_array
        self.cond_dict = nn.ModuleDict()
        
        for i, cond in enumerate(dim_conds):
            if cond is not None:
                self.cond_dict[f'cond_linear_{i}'] = nn.Linear(dim_conds[i], dim_modalities[i] * 2)

                self.cond_dict[f'to_post_branch_gammas_{i}'] = nn.Sequential(
                    Rearrange('b d -> b 1 d'),
                    nn.SiLU(),
                    self.cond_dict[f'cond_linear_{i}']
                )

                nn.init.zeros_(self.cond_dict[f'cond_linear_{i}'].weight)
                nn.init.constant_(self.cond_dict[f'cond_linear_{i}'].bias, 1.)

        # joint modality attention

        attention_layernorms = [AdaptiveLayerNorm(dim, dim_cond = dim_cond) for dim, dim_cond in zip(dim_modalities, dim_conds)]
        self.attn_layernorms = ModuleList(attention_layernorms)

        self.joint_attn = JointAttention(
            dim = dim_joint_attn,
            dim_inputs = dim_modalities,
            dim_head = dim_head,
            heads = heads,
            flash = flash_attn,
            qk_rmsnorm = qk_rmsnorm,
            softclamp = softclamp,
            softclamp_value = softclamp_value,
        )

        # feedforwards

        feedforward_layernorms = [AdaptiveLayerNorm(dim, dim_cond = dim_cond) for dim, dim_cond in zip(dim_modalities, dim_conds)]
        self.ff_layernorms = ModuleList(feedforward_layernorms)

        feedforwards = [FeedForward(dim, **ff_kwargs) for dim in dim_modalities]
        self.feedforwards = ModuleList(feedforwards)

    def forward(
        self,
        *,
        modality_tokens: Tuple[Tensor, ...],
        modality_masks: Tuple[Tensor | None, ...] | None = None,
        time_cond = Tuple[Tensor | None, ...]
    ):
        assert len(modality_tokens) == self.num_modalities and len(time_cond) == self.num_modalities

        attn_gammas = [1.] * len(time_cond) # Default to 1. if no condition
        ff_gammas = [1.] * len(time_cond) # Default to 1. if no condition
        for i, cond in enumerate(time_cond):
            if cond is not None:
                attn_gammas[i], ff_gammas[i] = self.cond_dict[f'to_post_branch_gammas_{i}'](cond).chunk(2, dim = -1)

        # attention layernorms
        modality_tokens_residual = list(modality_tokens)  # Create new list instead of modifying in place
        modality_tokens = [ln(tokens, cond) for tokens, cond, ln in zip(modality_tokens, time_cond, self.attn_layernorms)]

        # attention
        modality_tokens = self.joint_attn(inputs = modality_tokens, masks = modality_masks)

        # post attention gammas
        modality_tokens = [tokens * gamma for tokens, gamma in zip(modality_tokens, attn_gammas)]

        # add attention residual
        modality_tokens = [(tokens + residual) for tokens, residual in zip(modality_tokens, modality_tokens_residual)]

        # handle feedforward adaptive layernorm
        modality_tokens_residual = list(modality_tokens)
        modality_tokens = [ln(tokens, cond) for tokens, cond, ln in zip(modality_tokens, time_cond, self.ff_layernorms)]
        modality_tokens = [ff(tokens) for tokens, ff in zip(modality_tokens, self.feedforwards)]

        # post feedforward gammas
        modality_tokens = [tokens * gamma for tokens, gamma in zip(modality_tokens, ff_gammas)]

        # add feedforward residual
        modality_tokens = [(tokens + residual) for tokens, residual in zip(modality_tokens, modality_tokens_residual)]

        # returns

        return modality_tokens

# mm dit transformer - simply many blocks

class MMDiT(Module):
    def __init__(
        self,
        *,
        depth,
        dim_modalities: Tuple[int, ...],
        dim_conds: Tuple[int, ...],
        **block_kwargs
    ):
        super().__init__()
        blocks = [MMDiTBlock(dim_modalities = dim_modalities, dim_conds = dim_conds, **block_kwargs) for _ in range(depth)]
        self.blocks = ModuleList(blocks)

        norms = [RMSNorm(dim) for dim in dim_modalities]
        self.norms = ModuleList(norms)

    def forward(
        self,
        *,
        modality_tokens: Tuple[Tensor, ...],
        modality_masks: Tuple[Tensor | None, ...] | None = None,
        time_cond = Tuple[Tensor | None, ...]
    ):
        for block in self.blocks:
            modality_tokens = block(
                time_cond = time_cond,
                modality_tokens = modality_tokens,
                modality_masks = modality_masks
            )

        modality_tokens = [norm(tokens) for tokens, norm in zip(modality_tokens, self.norms)]

        return tuple(modality_tokens)

# New stuff

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t = t.flatten()
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, patch_size, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size,out_size, bias=True)
        self.adaLN_modulation = nn.Sequential(
            Rearrange('b d -> b 1 d'),
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = self.norm_final(x) * scale + shift
        x = self.linear(x)
        return x
    
def unpatchify(x, channels=3):
    patch_size = int((x.shape[2] // channels) ** 0.5)
    h = w = int(x.shape[1] ** .5)
    assert h * w == x.shape[1] and patch_size ** 2 * channels == x.shape[2]
    x = rearrange(x, 'B (h w) (p1 p2 C) -> B C (h p1) (w p2)', h=h, p1=patch_size, p2=patch_size)
    return x

class MMDiTModel(nn.Module, PyTorchModelHubMixin):
    def __init__(self, img_resolution, patch_size, img_channels, vocab_size, context_len, text_depth, image_depth, project_hidden=False, **kwargs):
        super().__init__()
        self.img_channels = img_channels
        self.dim_modalities = kwargs['dim_modalities']
        self.dim_conds = kwargs['dim_conds']
        self.dim_text = self.dim_modalities[0]
        self.dim_image = self.dim_modalities[1]

        assert img_resolution % patch_size == 0, f'img_resolution must be divisible by patch_size got {img_resolution} and {patch_size}'

        # Time Encoders
        self.text_time_encoder = TimestepEmbedder(self.dim_text)
        self.image_time_encoder = TimestepEmbedder(self.dim_image)

        # Text Embeddings
        self.text_embedder = nn.Embedding(vocab_size, self.dim_text)
        self.text_pos_embed = nn.Embedding(context_len, self.dim_text)
        nn.init.normal_(self.text_pos_embed.weight, std=0.02)

        # Image Embeddings
        self.image_embedder = PatchEmbed(patch_size, img_channels, self.dim_image)
        self.register_buffer('pos_embed', torch.from_numpy(get_2d_sincos_pos_embed(self.dim_image, img_resolution//patch_size)).float().unsqueeze(0))

        # Joint Embedding
        self.joint_embedding = MMDiT(**kwargs)

        # Single Embeddings
        self.has_text = text_depth > 0
        self.has_image = image_depth > 0
        sep_keys = ['depth', 'dim_modalities', 'dim_conds']
        block_kwargs = {k: v for k, v in kwargs.items() if k not in sep_keys}
        if self.has_text:
            self.text_dit = MMDiT(depth = text_depth, dim_modalities = [self.dim_text], dim_conds = [self.dim_text], **block_kwargs)
        else:
            self.freeze_last_block(0)
        if self.has_image:
            self.image_dit = MMDiT(depth = image_depth, dim_modalities = [self.dim_image], dim_conds = [self.dim_image], **block_kwargs)
        else:
            self.freeze_last_block(1)

        # Final Layers
        if self.has_text:
            self.text_final_layer = FinalLayer(self.dim_text, vocab_size)
        if self.has_image:
            self.image_final_layer = FinalLayer(self.dim_image, patch_size**2 * img_channels)

        # Hidden states projection 
        self.project_hidden = project_hidden
        if project_hidden:
            if self.has_text:
                self.text_hidden_projection = mlp(self.dim_text, self.dim_text, 768)

            if self.has_image:
                self.image_hidden_projection = mlp(self.dim_image, self.dim_image, 1024)

    def freeze_last_block(self, idx):
        last_block = self.joint_embedding.blocks[-1]
        feedforward = last_block.feedforwards[idx]
        layernorm = last_block.ff_layernorms[idx]
        last_block.joint_attn.to_out[idx].weight.requires_grad = False
        cond_block = last_block.cond_dict[f'cond_linear_{idx}']
        self.joint_embedding.norms[idx].g.requires_grad = False
        for param in feedforward.parameters():
            param.requires_grad = False
        for param in layernorm.parameters():
            param.requires_grad = False
        for param in cond_block.parameters():
            param.requires_grad = False
    
    def freeze_joint(self):
        layers_to_freeze = [self.text_embedder, self.text_pos_embed, self.text_time_encoder, 
                            self.image_embedder, self.image_time_encoder,
                            self.joint_embedding]
        for layer in layers_to_freeze:
            for param in layer.parameters():
                param.requires_grad = False

    def freeze_image(self):
        layers_to_freeze = [self.image_embedder, self.image_time_encoder, self.image_dit, self.image_final_layer]
        for layer in layers_to_freeze:
            for param in layer.parameters():
                param.requires_grad = False
    
    def freeze_text(self):
        layers_to_freeze = [self.text_embedder, self.text_pos_embed, self.text_time_encoder, self.text_dit, self.text_final_layer]
        for layer in layers_to_freeze:
            for param in layer.parameters():
                param.requires_grad = False

    def forward(
        self,
        *,
        text_tokens,
        image,
        text_mask = None,
        image_time_cond = None,
        text_time_cond = None,
        detach_hidden = [False, False]
    ):
        # Create position indices based on sequence length
        positions = torch.arange(text_tokens.shape[1], device=text_tokens.device)
        positions = positions.unsqueeze(0).expand(text_tokens.shape[0], -1)[:, :text_tokens.shape[1]] # [batch, seq_len]
        
        image_tokens = self.image_embedder(image) + self.pos_embed
        text_tokens = self.text_embedder(text_tokens) + self.text_pos_embed(positions)
        text_time_cond = self.text_time_encoder(text_time_cond)
        image_time_cond = self.image_time_encoder(image_time_cond)

        text_tokens_hidden, image_tokens_hidden = self.joint_embedding(
            modality_tokens = (text_tokens, image_tokens),
            modality_masks = (text_mask, None),
            time_cond = (text_time_cond, image_time_cond),
        )
        if detach_hidden[0]:
            text_tokens_hidden = text_tokens_hidden.detach()
        if detach_hidden[1]:
            image_tokens_hidden = image_tokens_hidden.detach()

        if self.has_text:
            text_tokens = self.text_dit(
                modality_tokens = (text_tokens_hidden,),
                modality_masks = (text_mask,),
                time_cond = (text_time_cond,),
            )[0]
            text_tokens = self.text_final_layer(text_tokens, text_time_cond)
            text_tokens[:, :, :-1] = text_tokens[:, :, :-1].log_softmax(dim=-1)

        if self.has_image:
            image_tokens = self.image_dit(
                modality_tokens = (image_tokens_hidden,),
                modality_masks = (None,),
                time_cond = (image_time_cond,),
            )[0]

            image_tokens = self.image_final_layer(image_tokens, image_time_cond)
            image_tokens = unpatchify(image_tokens, self.img_channels)


        # Project hidden states
        if self.project_hidden:
            text_tokens_hidden = self.text_hidden_projection(text_tokens_hidden) if self.has_text else None
            image_tokens_hidden = self.image_hidden_projection(image_tokens_hidden) if self.has_image else None
        else:
            image_tokens_hidden = None
            text_tokens_hidden = None

        return image_tokens, text_tokens

