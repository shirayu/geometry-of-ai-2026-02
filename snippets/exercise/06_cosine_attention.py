import torch
import torch.nn.functional as F
import math


def standard_attention(Q, K, V):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, V), weights


def cosine_attention(Q, K, V, scale=10.0):
    Q_norm = F.normalize(Q, dim=-1)
    K_norm = F.normalize(K, dim=-1)
    scores = torch.matmul(Q_norm, K_norm.transpose(-2, -1)) * scale
    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, V), weights


# テスト
Q = torch.randn(1, 4, 8)  # batch=1, seq=4, dim=8
K = torch.randn(1, 4, 8)
V = torch.randn(1, 4, 8)

out_std, w_std = standard_attention(Q, K, V)
out_cos, w_cos = cosine_attention(Q, K, V)

print("Standard weights:\n", w_std[0].round(decimals=3))
print("Cosine weights:\n", w_cos[0].round(decimals=3))
