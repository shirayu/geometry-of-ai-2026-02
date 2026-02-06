import time

import torch


def standard_attention(Q, K, V):
    """標準的なAttention（密な計算、O(n²)メモリ）"""
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k**0.5)
    attention_weights = torch.softmax(scores, dim=-1)  # O(n^2) のメモリ
    output = torch.matmul(attention_weights, V)
    return output


def topk_reweighting_attention(Q, K, V, k=10):
    """Top-Kによる情報的重み付け（教育目的の概念実装）

    注意：このコードは計算を省略していない（全スコアを計算している）。
    真の計算的剪定には、スコア計算自体をスキップする必要がある。
    実行速度は標準実装と同等か、むしろ遅い可能性が高い。
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k**0.5)  # ← 全計算（省略なし）

    # Top-K選択（各クエリについて上位k個のキーのみ保持）
    topk_scores, topk_indices = torch.topk(scores, k=k, dim=-1)

    # スパース行列を構築（実際にはより効率的な実装が必要）
    sparse_weights = torch.zeros_like(scores)  # ← ゼロ行列作成（非効率）
    sparse_weights.scatter_(-1, topk_indices, torch.softmax(topk_scores, dim=-1))

    output = torch.matmul(sparse_weights, V)
    return output


# ベンチマーク（参考：速度比較は意味を持たない）
batch_size, num_heads, seq_len, d_k = 2, 8, 512, 64
Q = torch.randn(batch_size, num_heads, seq_len, d_k, device="cuda")
K = torch.randn(batch_size, num_heads, seq_len, d_k, device="cuda")
V = torch.randn(batch_size, num_heads, seq_len, d_k, device="cuda")

# 標準Attention
start = time.time()
out1 = standard_attention(Q, K, V)
torch.cuda.synchronize()
time1 = time.time() - start

# Top-K重み付け (k=32) - 教育目的実装
start = time.time()
out2 = topk_reweighting_attention(Q, K, V, k=32)
torch.cuda.synchronize()
time2 = time.time() - start

print(f"Standard Attention: {time1:.4f}s")
print(f"Top-K Reweighting (k=32, 教育実装): {time2:.4f}s")
print("Note: 教育実装は標準実装と同等か、むしろ遅い可能性が高い")
print("      真の高速化には専用カーネルが必要")
