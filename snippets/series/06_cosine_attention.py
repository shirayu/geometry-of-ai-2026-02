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


def cosine_attention(Q, K, V, temperature=1.0, mask=None):
    """Cosine Attention（Q, Kを正規化）

    この設計では、Q^T K = cos(θ) が厳密に成り立つ。

    Args:
        Q, K, V: Query, Key, Value
        temperature: 温度パラメータ（大きいほど分布が平坦）
        mask: オプションのマスク

    Returns:
        output, attention_weights
    """
    # Q, Kを単位ノルムに正規化
    Q_norm = F.normalize(Q, dim=-1)  # dim=-1 で最後の次元を正規化
    K_norm = F.normalize(K, dim=-1)

    # 内積 = コサイン類似度
    scores = torch.matmul(Q_norm, K_norm.transpose(-2, -1)) / temperature

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    attention_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, V)

    return output, attention_weights


# 使用例：標準 vs Cosine の比較
batch, heads, seq_len, d_k = 2, 8, 10, 64
Q = torch.randn(batch, heads, seq_len, d_k)
K = torch.randn(batch, heads, seq_len, d_k)
V = torch.randn(batch, heads, seq_len, d_k)

# 標準Attention
out_standard, attn_standard = scaled_dot_product_attention(Q, K, V)

# Cosine Attention
out_cosine, attn_cosine = cosine_attention(Q, K, V)

print("標準Attention: scores range depends on norms")
print("Cosine Attention: scores in [-1, 1] (cosine similarity)")
