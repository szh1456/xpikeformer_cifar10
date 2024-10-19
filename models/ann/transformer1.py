# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from models.ann.utils import Conv1D, NewGELUActivation

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class ViT(nn.Module):
    def __init__(
        self, *, paradigm, model_name, encoding, image_size, patch_size, num_classes,
        dim, depth, heads, pool = 'cls', channels = 3, dim_head = 64,
        dropout = 0., emb_dropout = 0., is_sto=False):
        super().__init__()
        self.paradigm = paradigm
        self.model_name = model_name
        self.encoding = encoding
        self.image_size = image_size
        self.patch_size = patch_size
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.n_class = num_classes
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        self.n_positions = num_patches + 1
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.quant = torch.ao.quantization.QuantStub()

        self.to_patch_embedding = SPT(dim = dim, patch_size = patch_size, channels = channels)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim)) # (1, 1, 1, d)

        self._backbone = TransformerBackbone(
            n_positions=self.n_positions,
            n_embd=self.dim,
            n_layer=self.depth,
            n_head=self.heads,
            resid_pdrop=dropout,
            embd_pdrop=dropout,
            attn_pdrop=dropout
        )

        self.pool = pool
        self._read_out = nn.Linear(self.dim, self.n_class)
        self.dequant = torch.ao.quantization.DeQuantStub()


    def forward(self, img):
        # (t, b, c, h, w) -> (t, b, n, d)
        img = self.quant(img)
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        # (1, 1, d) -> (b, 1, d)
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        # (b, 1, d) cat (b, n, d) -> (b, n+1, d)
        x = torch.cat((cls_tokens, x), dim=1)

        # position_ids = torch.arange(0, self.n_positions, dtype=torch.long, device=x.device)
        # pos_embed = self.pos_embedding(position_ids)
        # x += pos_embed
        # x = self.dropout(x)
        output = self._backbone(x)
        prediction = self._read_out(output)
        prediction = self.dequant(prediction)
        return prediction[:, 0, :]

class SPT(nn.Module):
    def __init__(self, *, dim, patch_size, channels = 3):
        super().__init__()
        patch_dim = patch_size * patch_size * 5 * channels

        self.to_patch_tokens = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_dim, dim)
        )

    def forward(self, x):
        shifts = ((1, -1, 0, 0), (-1, 1, 0, 0), (0, 0, 1, -1), (0, 0, -1, 1))
        shifted_x = list(map(lambda shift: F.pad(x, shift), shifts))
        x_with_shifts = torch.cat((x, *shifted_x), dim = 1)
        return self.to_patch_tokens(x_with_shifts)

class TransformerBackbone(nn.Module):
    def __init__(self, n_positions, n_embd, n_layer, n_head, resid_pdrop, embd_pdrop, attn_pdrop=0.0):
        super().__init__()
        self.embd_dim = n_embd
        self.n_positions = n_positions

        self.pos_embedding = nn.Embedding(n_positions, self.embd_dim)
        self.dropout = nn.Dropout(embd_pdrop)

        self.h = nn.ModuleList([TransformerBlock(n_positions, n_embd, n_head, resid_pdrop, attn_pdrop) for i in range(n_layer)])

        # self.ln_f = nn.LayerNorm(self.embd_dim)

    def forward(self, inputs_embeds):
        position_ids = torch.arange(0, self.n_positions, dtype=torch.long, device=inputs_embeds.device)
        pos_embed = self.pos_embedding(position_ids)
        hidden_states = inputs_embeds + pos_embed
        hidden_states = self.dropout(hidden_states)
        for layer in self.h:
            hidden_states = layer(hidden_states)
        # hidden_states = self.ln_f(hidden_states)

        return hidden_states


class TransformerBlock(nn.Module):
    def __init__(self, n_positions, hidden_dim, n_head, resid_pdrop,  attn_pdrop=0.):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attn = TransformerAttention(n_positions, hidden_dim,n_head, resid_pdrop, attn_pdrop)
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.ff = TransformerFeedForward(hidden_dim, resid_pdrop)
        self.ln_2 = nn.LayerNorm(hidden_dim)

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        hidden_states = self.attn(hidden_states) + residual
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        hidden_states = self.ff(hidden_states) + residual

        return hidden_states

class TransformerAttention(nn.Module):
    def __init__(self, n_positions,hidden_dim, n_head, resid_pdrop, attn_pdrop = 0.):
        super().__init__()
        self.embed_dim = hidden_dim
        self.n_head = n_head
        self.head_dim = self.embed_dim // self.n_head
        self.split_size = self.embed_dim
        if self.head_dim * self.n_head != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by n_head (got `embed_dim`: {self.embed_dim} and `n_head`:"
                f" {self.n_head})."
            )
        self_mask = ~torch.eye(n_positions, dtype=torch.bool)
        self.register_buffer("self_mask", self_mask.view(1, 1, n_positions, n_positions))

        self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)
        self.out = nn.Linear(self.embed_dim,self.embed_dim)
        self.attn_dropout = nn.Dropout(attn_pdrop)
        # self.resid_dropout = nn.Dropout(resid_pdrop)

    def _attn(self, query, key, value):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))
        attn_weights = attn_weights / torch.full(
                [], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
            )
        mask_value = torch.finfo(attn_weights.dtype).min
        attn_weights = torch.where(self.self_mask, attn_weights.to(attn_weights.dtype), mask_value)
        
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights.type(value.dtype)
        attn_output = torch.matmul(attn_weights, value)
        return attn_output
    
    def _split_heads(self, tensor, n_head, attn_head_size):
        # [batch, sequence_len, embeded_dim] -> [batch, heads, sequence_len, head_dim]
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (n_head, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)
    
    def _merge_heads(self, tensor, n_head, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (n_head * attn_head_size,)
        return tensor.view(new_shape)
    
    def forward(self, hidden_states):
        query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
        # [batch, sequence_len, embeded_dim] -> [batch, heads, sequence_len, head_dim]
        query = self._split_heads(query, self.n_head, self.head_dim)
        key = self._split_heads(key, self.n_head, self.head_dim)
        value = self._split_heads(value, self.n_head, self.head_dim)
        attn_output = self._attn(query, key, value)
        attn_output = self._merge_heads(attn_output, self.n_head, self.head_dim)
        attn_output = self.out(attn_output)
        attn_output = self.attn_dropout(attn_output)
        return attn_output


class TransformerFeedForward(nn.Module):
    def __init__(self, hidden_dim, ff_pdrop):
        super().__init__()
        embed_dim = hidden_dim
        intermediate_dim = 4 * hidden_dim
        self.c_fc = Conv1D(intermediate_dim, embed_dim)
        self.c_proj = Conv1D(embed_dim, intermediate_dim)
        self.act = NewGELUActivation()
        self.dropout = nn.Dropout(ff_pdrop)

    def forward(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states
