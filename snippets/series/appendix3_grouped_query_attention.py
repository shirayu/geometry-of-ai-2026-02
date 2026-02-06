import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupedQueryAttention(nn.Module):
    """GQA (Grouped-Query Attention) の実装"""

    def __init__(self, d_model, num_query_heads, num_kv_heads):
        super().__init__()
        assert num_query_heads % num_kv_heads == 0, "num_query_heads must be divisible by num_kv_heads"

        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.d_k = d_model // num_query_heads
        self.group_size = num_query_heads // num_kv_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, self.num_kv_heads * self.d_k)
        self.W_v = nn.Linear(d_model, self.num_kv_heads * self.d_k)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # Q: [batch, seq_len, num_query_heads, d_k]
        Q = self.W_q(x).view(batch_size, seq_len, self.num_query_heads, self.d_k).transpose(1, 2)

        # K, V: [batch, seq_len, num_kv_heads, d_k]
        K = self.W_k(x).view(batch_size, seq_len, self.num_kv_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_kv_heads, self.d_k).transpose(1, 2)

        # K, Vを各グループで共有するため、repeatで拡張
        # [batch, num_kv_heads, seq_len, d_k] -> [batch, num_query_heads, seq_len, d_k]
        K = K.repeat_interleave(self.group_size, dim=1)
        V = V.repeat_interleave(self.group_size, dim=1)

        # 通常のAttention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k**0.5)
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)

        # ヘッドを結合
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.W_o(output)


# 使用例: 8つのQueryヘッド、2つのKVヘッド（4:1のグループ化）
gqa = GroupedQueryAttention(d_model=512, num_query_heads=8, num_kv_heads=2)
x = torch.randn(2, 10, 512)
out = gqa(x)
print(f"GQA output shape: {out.shape}")
print(f"Parameters saved: ~{(1 - 2 / 8) * 100:.1f}% (for K,V)")
