import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def precompute_rope_freqs(dim, max_seq_len, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    positions = torch.arange(max_seq_len).float()
    angles = torch.outer(positions, freqs)
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x, freqs_cos, freqs_sin):
    x_reshape = x.reshape(*x.shape[:-1], -1, 2)
    x1, x2 = x_reshape[..., 0], x_reshape[..., 1]
    seq_len = x.size(-2)
    cos = freqs_cos[:seq_len].unsqueeze(0).unsqueeze(0)
    sin = freqs_sin[:seq_len].unsqueeze(0).unsqueeze(0)
    x1_rot = x1 * cos - x2 * sin
    x2_rot = x1 * sin + x2 * cos
    x_rot = torch.stack([x1_rot, x2_rot], dim=-1)
    return x_rot.reshape(*x.shape)


def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, V)
    return output, attention_weights


def cosine_attention(Q, K, V, temperature=1.0, mask=None):
    Q_norm = F.normalize(Q, dim=-1)
    K_norm = F.normalize(K, dim=-1)
    scores = torch.matmul(Q_norm, K_norm.transpose(-2, -1)) / temperature
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, V)
    return output, attention_weights


class MultiHeadAttention(nn.Module):
    """Multi-head Attention（教育目的の実装）"""

    def __init__(self, d_model, num_heads, use_rope=False, use_cosine=False):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.use_rope = use_rope
        self.use_cosine = use_cosine

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        if use_rope:
            freqs_cos, freqs_sin = precompute_rope_freqs(self.d_k, 2048)
            self.register_buffer("freqs_cos", freqs_cos)
            self.register_buffer("freqs_sin", freqs_sin)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        # 線形射影
        Q = (
            self.W_q(x)
            .view(batch_size, seq_len, self.num_heads, self.d_k)
            .transpose(1, 2)
        )
        K = (
            self.W_k(x)
            .view(batch_size, seq_len, self.num_heads, self.d_k)
            .transpose(1, 2)
        )
        V = (
            self.W_v(x)
            .view(batch_size, seq_len, self.num_heads, self.d_k)
            .transpose(1, 2)
        )

        # RoPEの適用（オプション）
        if self.use_rope:
            Q = apply_rope(Q, self.freqs_cos, self.freqs_sin)
            K = apply_rope(K, self.freqs_cos, self.freqs_sin)

        # Attention計算
        if self.use_cosine:
            output, _ = cosine_attention(Q, K, V, mask=mask)
        else:
            output, _ = scaled_dot_product_attention(Q, K, V, mask=mask)

        # ヘッドを結合
        output = (
            output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        )

        return self.W_o(output)
