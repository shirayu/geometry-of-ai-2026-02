import math

import torch
import torch.nn.functional as F


def scaled_dot_product_attention(Q, K, V, mask=None):
    """標準的なScaled Dot-Product Attention

    Args:
        Q: Query [batch, heads, seq_len, d_k]
        K: Key [batch, heads, seq_len, d_k]
        V: Value [batch, heads, seq_len, d_v]
        mask: オプションのマスク [batch, 1, 1, seq_len] or [batch, 1, seq_len, seq_len]

    Returns:
        output: Attention出力 [batch, heads, seq_len, d_v]
        attention_weights: Attention重み [batch, heads, seq_len, seq_len]
    """
    d_k = Q.size(-1)

    # Q @ K^T / sqrt(d_k)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    # マスクの適用（オプション）
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    # Softmaxで確率に変換
    attention_weights = F.softmax(scores, dim=-1)

    # Value との加重和
    output = torch.matmul(attention_weights, V)

    return output, attention_weights
