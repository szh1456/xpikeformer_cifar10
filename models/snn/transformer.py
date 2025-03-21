import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from spikingjelly.activation_based import layer, neuron, functional
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from .bneuron import Bernoulli_neuron

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class SViT(nn.Module):
    def __init__(
        self, *, paradigm, model_name, encoding, image_size, patch_size, num_classes,
        dim, depth, heads, channels = 3, dim_head = 64,
        dropout = 0., emb_dropout = 0.):
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
        assert image_height % patch_height == 0 and image_width % patch_width == 0 # Image dimensions must be divisible by the patch size.
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        self.n_positions = num_patches + 1

        self.quant = torch.ao.quantization.QuantStub()

        self.to_patch_embedding = SSPT(dim = dim, patch_size = patch_size, channels = channels)
        self.cls_token = nn.Parameter(torch.randn(1, 1, 1, dim))

        self._backbone = TransformerBackbone(
            n_positions=self.n_positions,
            n_embd=self.dim,
            n_layer=self.depth,
            n_head=self.heads,
            resid_pdrop=dropout,
            embd_pdrop=dropout,
            attn_pdrop=dropout
        )

        self._read_out = nn.Linear(self.dim, self.n_class)
        self.dequant = torch.ao.quantization.DeQuantStub()


    def forward(self, img):
        # (t, b, c, h, w) -> (t, b, n, d)
        img = self.quant(img)
        x = self.to_patch_embedding(img)
        t, b, n, _ = x.shape

        # (1, 1, 1, d) -> (t, b, 1, d)
        cls_tokens = repeat(self.cls_token, '() () () d -> t b () d', t=t, b=b)
        # (t, b, 1, d) cat (t, b, n, d) -> (t, b, n+1, d)
        x = torch.cat((cls_tokens, x), dim=2)

        # position_ids = torch.arange(0, self.n_positions, dtype=torch.long, device=x.device)
        # pos_embed = self.pos_embedding(position_ids)
        # x += pos_embed
        # x = self.dropout(x)
        output = self._backbone(x)
        prediction = self._read_out(output)
        prediction = torch.mean(prediction, dim=0)
        prediction = self.dequant(prediction)
        return prediction[:, 0, :]

class SSPT(nn.Module):
    def __init__(self, *, dim, patch_size, channels = 3):
        super().__init__()
        patch_dim = patch_size * patch_size * 5 * channels

        self.to_patch_tokens = nn.Sequential(
            Rearrange('t b c (h p1) (w p2) -> t b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim)
        )

    def forward(self, x):
        shifts = ((1, -1, 0, 0), (-1, 1, 0, 0), (0, 0, 1, -1), (0, 0, -1, 1))
        shifted_x = list(map(lambda shift: F.pad(x, shift), shifts))
        x_with_shifts = torch.cat((x, *shifted_x), dim = 2)
        return self.to_patch_tokens(x_with_shifts)

class TransformerBackbone(nn.Module):
    def __init__(self, n_positions, n_embd, n_layer, n_head, resid_pdrop, embd_pdrop, attn_pdrop=0.0):
        super().__init__()
        self.embd_dim = n_embd
        self.n_positions = n_positions

        self.pos_embedding = nn.Embedding(n_positions, self.embd_dim)
        self.embedding_lif = neuron.LIFNode(tau=2., decay_input=False, v_threshold=1., step_mode='m',backend='cupy')
        self.dropout = layer.Dropout(embd_pdrop, step_mode='m')

        self.h = nn.ModuleList(
            [TransformerBlock(n_positions, n_embd, n_head, resid_pdrop, attn_pdrop) for i in range(n_layer)])

    def forward(self, inputs_embeds):
        position_ids = torch.arange(0, self.n_positions, dtype=torch.long, device=inputs_embeds.device)
        pos_embed = self.pos_embedding(position_ids)
        hidden_states = inputs_embeds + pos_embed
        # hidden_states = self.embedding_lif(hidden_states)
        hidden_states = self.dropout(hidden_states)
        for layer in self.h:
            hidden_states = layer(hidden_states)
        # hidden_states = self.ln_f(hidden_states)

        return hidden_states


class TransformerBlock(nn.Module):
    def __init__(self, n_positions, hidden_dim, n_head, resid_pdrop, attn_pdrop=0.):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attn = TransformerAttention(n_positions, hidden_dim, n_head, resid_pdrop, attn_pdrop)
        self.ff = TransformerFeedForward(hidden_dim, resid_pdrop)

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.attn(hidden_states) + residual
        residual = hidden_states
        hidden_states = self.ff(hidden_states) + residual

        return hidden_states

class TransformerAttention(nn.Module):
    def __init__(self, n_positions, hidden_dim, n_head, resid_pdrop, attn_pdrop=0.):
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

        self.c_attn = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=False)
        self.c_bn = nn.BatchNorm1d(3 * self.embed_dim)
        self.c_attn_lif = neuron.LIFNode(tau=2., decay_input=False, v_threshold=1., step_mode='m',backend='cupy')
        'pre_train with standard LIF or IF'
        # self.attn_weight_lif = neuron.LIFNode(tau=1.1, decay_input=False, v_threshold=0.5, step_mode='m',backend='cupy')
        # self.attn_output_lif = neuron.LIFNode(tau=1.1, decay_input=False, v_threshold=0.5, step_mode='m',backend='cupy')
        'fine tune with Bernoulli_neuron'
        # self.attn_weight_lif = Bernoulli_neuron(step_mode='m',backend='cupy',v_reset=None)
        # self.attn_output_lif = Bernoulli_neuron(step_mode='m',backend='cupy',v_reset=None)

        self.attn_dropout = layer.Dropout(attn_pdrop, step_mode='m')
        self.resid_dropout = layer.Dropout(resid_pdrop, step_mode='m')
        self.attn_output_linear = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.attn_output_bn = nn.BatchNorm1d(self.embed_dim)

    def _attn(self, query, key, value):
        t,bs,nh,l_seq,head_dim = query.shape
        attn_weights = torch.matmul(query, key.transpose(-1, -2))  #(t,bs,nh,l_seq,head_dim)
        attn_weights = torch.bernoulli((attn_weights/16).clamp(0.,1.))
        mask_value = torch.tensor(.0, dtype=attn_weights.dtype).to(attn_weights.device)
        attn_weights = torch.where(self.self_mask, attn_weights.to(attn_weights.dtype), mask_value)
        attn_output = torch.matmul(attn_weights, value)
        attn_output = torch.bernoulli((attn_output/16).clamp(0.,1.))
        return attn_output

    def _split_heads(self, tensor, n_head, attn_head_size):
        # [T, batch, sequence_len, embeded_dim] -> [T, batch, heads, sequence_len, head_dim]
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (n_head, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 1, 3, 2, 4)

    def _merge_heads(self, tensor, n_head, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tensor.permute(0, 1, 3, 2, 4).contiguous()
        new_shape = tensor.size()[:-2] + (n_head * attn_head_size,)
        return tensor.view(new_shape)

    def forward(self, hidden_states):
        T,B,N,D = hidden_states.shape
        # print(hidden_states[0,0,:,:])
        hidden_states = hidden_states.flatten(0,1)
        hidden_states = self.c_attn(hidden_states)
        hidden_states = self.c_bn(hidden_states.transpose(-1,-2)).transpose(-1, -2).reshape(T, B, N, 3*D).contiguous()
        hidden_states = self.c_attn_lif(hidden_states)
        query, key, value = hidden_states.split(self.split_size, dim=-1)
        # [T, batch, sequence_len, embeded_dim] -> [T, batch, heads, sequence_len, head_dim]
        query = self._split_heads(query, self.n_head, self.head_dim)
        key = self._split_heads(key, self.n_head, self.head_dim)
        value = self._split_heads(value, self.n_head, self.head_dim)
        attn_output = self._attn(query, key, value)
        attn_output = self._merge_heads(attn_output, self.n_head, self.head_dim)
        attn_output = self.attn_output_bn(attn_output.flatten(0,1).transpose(-1,-2)).transpose(-1, -2).reshape(T, B, N, D).contiguous()
        attn_output = self.attn_output_linear(attn_output)
        return attn_output


class TransformerFeedForward(nn.Module):
    def __init__(self, hidden_dim, ff_pdrop):
        super().__init__()
        embed_dim = hidden_dim
        intermediate_dim = 4 * hidden_dim
        self.fc = nn.Linear(embed_dim, intermediate_dim, bias=False)
        self.act = neuron.LIFNode(tau=1.2, decay_input=False, v_threshold=1., step_mode='m',backend='cupy')
        self.proj = nn.Linear(intermediate_dim, embed_dim, bias=False)
        self.ln = nn.LayerNorm(embed_dim)
        self.dropout = layer.Dropout(ff_pdrop, step_mode='m')

    def forward(self, hidden_states):
        hidden_states = self.fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.proj(hidden_states)
        hidden_states = self.ln(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

class SPS(nn.Module):
    def __init__(self, img_size_h=32, img_size_w=32, patch_size=4, in_channels=3, embed_dims=512):
        super().__init__()
        self.image_size = [img_size_h, img_size_w]
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.C = in_channels
        self.H, self.W = self.image_size[0] // patch_size[0], self.image_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj_conv = nn.Conv2d(in_channels, embed_dims//8, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(embed_dims//8)
        self.proj_lif = neuron.LIFNode(tau=2.0, detach_reset=True, backend='cupy', step_mode='m')

        self.proj_conv1 = nn.Conv2d(embed_dims//8, embed_dims//4, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn1 = nn.BatchNorm2d(embed_dims//4)
        self.proj_lif1 = neuron.LIFNode(tau=2.0, detach_reset=True, backend='cupy', step_mode='m')

        self.proj_conv2 = nn.Conv2d(embed_dims//4, embed_dims//2, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn2 = nn.BatchNorm2d(embed_dims//2)
        self.proj_lif2 = neuron.LIFNode(tau=2.0, detach_reset=True, backend='cupy', step_mode='m')
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.proj_conv3 = nn.Conv2d(embed_dims//2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn3 = nn.BatchNorm2d(embed_dims)
        self.proj_lif3 = neuron.LIFNode(tau=2.0, detach_reset=True, backend='cupy', step_mode='m')
        self.maxpool3 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.rpe_conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.rpe_bn = nn.BatchNorm2d(embed_dims)
        self.rpe_lif = neuron.LIFNode(tau=2.0, detach_reset=True, backend='cupy', step_mode='m')

    def forward(self, x):
        # (t, b, c, h, w)
        T, B, C, H, W = x.shape
        x = self.proj_conv(x.flatten(0, 1)) # have some fire value # (tb, c, h, w)
        x = self.proj_bn(x).reshape(T, B, -1, H, W).contiguous() # (t, b, c, h, w)
        x = self.proj_lif(x).flatten(0, 1).contiguous() # (tb, c, h, w)

        x = self.proj_conv1(x) # (tb, c, h, w)
        x = self.proj_bn1(x).reshape(T, B, -1, H, W).contiguous()
        x = self.proj_lif1(x).flatten(0, 1).contiguous()  # (tb, c, h, w)

        x = self.proj_conv2(x)  # (tb, c, h, w)
        x = self.proj_bn2(x).reshape(T, B, -1, H, W).contiguous() # (t, b, c, h, w)
        x = self.proj_lif2(x).flatten(0, 1).contiguous() # (tb, c, h, w)
        x = self.maxpool2(x)

        x = self.proj_conv3(x)
        x = self.proj_bn3(x).reshape(T, B, -1, H//2, W//2).contiguous()
        x = self.proj_lif3(x).flatten(0, 1).contiguous()
        x = self.maxpool3(x)

        x_feat = x.reshape(T, B, -1, H//4, W//4).contiguous()
        x = self.rpe_conv(x)
        x = self.rpe_bn(x).reshape(T, B, -1, H//4, W//4).contiguous()
        x = self.rpe_lif(x)
        x = x + x_feat

        x = x.flatten(-2).transpose(-1, -2)  # T,B,N,C
        return x