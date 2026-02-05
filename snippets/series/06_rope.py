import math

import torch
import torch.nn.functional as F


def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, V)
    return output, attention_weights


def precompute_rope_freqs(dim, max_seq_len, theta=10000.0):
    """RoPEの周波数を事前計算

    Args:
        dim: 埋め込み次元（偶数）
        max_seq_len: 最大シーケンス長
        theta: 基本周波数

    Returns:
        freqs_cos, freqs_sin: [max_seq_len, dim/2] の周波数テンソル
    """
    # 各次元ペアの周波数
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))

    # 位置インデックス
    positions = torch.arange(max_seq_len).float()

    # 外積で [seq_len, dim/2] の角度を計算
    angles = torch.outer(positions, freqs)

    return torch.cos(angles), torch.sin(angles)


def apply_rope(x, freqs_cos, freqs_sin):
    """RoPEを適用

    Args:
        x: 入力テンソル [batch, heads, seq_len, dim]
        freqs_cos, freqs_sin: 事前計算された周波数

    Returns:
        回転が適用されたテンソル
    """
    # 次元を2つずつのペアに分割
    x_reshape = x.reshape(*x.shape[:-1], -1, 2)
    x1, x2 = x_reshape[..., 0], x_reshape[..., 1]

    # シーケンス長に合わせてスライス
    seq_len = x.size(-2)
    cos = freqs_cos[:seq_len].unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, dim/2]
    sin = freqs_sin[:seq_len].unsqueeze(0).unsqueeze(0)

    # 回転を適用: (x1, x2) -> (x1*cos - x2*sin, x1*sin + x2*cos)
    x1_rot = x1 * cos - x2 * sin
    x2_rot = x1 * sin + x2 * cos

    # 元の形状に戻す
    x_rot = torch.stack([x1_rot, x2_rot], dim=-1)
    return x_rot.reshape(*x.shape)


# 使用例
dim, max_len = 64, 512
freqs_cos, freqs_sin = precompute_rope_freqs(dim, max_len)
Q = torch.randn(2, 8, 10, dim)
K = torch.randn(2, 8, 10, dim)
V = torch.randn(2, 8, 10, dim)

# Query, Keyに適用
Q_rope = apply_rope(Q, freqs_cos, freqs_sin)
K_rope = apply_rope(K, freqs_cos, freqs_sin)

# RoPE適用後のAttention
out_rope, attn_rope = scaled_dot_product_attention(Q_rope, K_rope, V)
