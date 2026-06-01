import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
from torch import einsum

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d


class CausalSelfAttention(nn.Module):
    """因果自注意力：每个时间步只能看到≤自己位置，内置因果mask"""
    def __init__(self, embed_dim, num_heads, dropout=0.1, bias=False):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def positional_encoding(self, seq_len, d_model):
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x, return_attention=False):
        N, L, D = x.size()
        pos_enc = self.positional_encoding(L, D)
        x = x + pos_enc.to(x.device)

        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(N, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # 因果mask：屏蔽未来位置（上三角为-inf）
        causal_mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
        attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).reshape(N, L, D)
        output = self.out_proj(output)
        output = self.resid_dropout(output)

        if return_attention:
            return output, attn_weights
        return output


class CausalCrossAttention(nn.Module):
    """因果交叉注意力：q位置i只能attend k位置≤i，内置因果mask"""
    def __init__(self, *, dim, heads, dim_head, context_dim=None, dropout=0.1,
                 talking_heads=False, prenorm=False):
        super().__init__()
        context_dim = default(context_dim, dim)
        self.norm = nn.LayerNorm(dim) if prenorm else nn.Identity()
        self.context_norm = nn.LayerNorm(context_dim) if prenorm else nn.Identity()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads
        self.dropout = nn.Dropout(dropout)
        self.context_dropout = nn.Dropout(dropout)
        self.to_qk = nn.Linear(dim, inner_dim, bias=False)
        self.context_to_qk = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.context_to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.context_to_out = nn.Linear(inner_dim, context_dim)
        self.talking_heads = nn.Conv2d(heads, heads, 1, bias=False) if talking_heads else nn.Identity()
        self.context_talking_heads = nn.Conv2d(heads, heads, 1, bias=False) if talking_heads else nn.Identity()

    def forward(self, x, context, mask=None, context_mask=None, return_attn=False, rel_pos_bias=None):
        b, i, j, h, device = x.shape[0], x.shape[-2], context.shape[-2], self.heads, x.device

        x = self.norm(x)
        context = self.context_norm(context)

        qk, v = self.to_qk(x), self.to_v(x)
        context_qk, context_v = self.context_to_qk(context), self.context_to_v(context)

        qk, context_qk, v, context_v = map(
            lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h),
            (qk, context_qk, v, context_v)
        )

        sim = einsum('b h i d, b h j d -> b h i j', qk, context_qk) * self.scale

        if exists(rel_pos_bias):
            sim = sim + rel_pos_bias

        # 因果mask：位置i只能attend到≤i的位置
        causal_mask = torch.triu(torch.ones(i, j, device=device, dtype=torch.bool), diagonal=1)
        sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        if exists(mask) or exists(context_mask):
            mask = default(mask, torch.ones((b, i), device=device, dtype=torch.bool))
            context_mask = default(context_mask, torch.ones((b, j), device=device, dtype=torch.bool))
            attn_mask = rearrange(mask, 'b i -> b 1 i 1') * rearrange(context_mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~attn_mask, -torch.finfo(sim.dtype).max)

        attn = sim.softmax(dim=-1)
        context_sim = sim.transpose(-1, -2)
        context_attn = context_sim.softmax(dim=-1)

        attn = self.dropout(attn)
        context_attn = self.context_dropout(context_attn)

        attn = self.talking_heads(attn)
        context_attn = self.context_talking_heads(context_attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, context_v)
        context_out = einsum('b h j i, b h i d -> b h j d', context_attn, v)

        out, context_out = map(lambda t: rearrange(t, 'b h n d -> b n (h d)'), (out, context_out))

        out = self.to_out(out)
        context_out = self.context_to_out(context_out)

        if return_attn:
            return out, context_out, attn, context_attn
        return out, context_out
