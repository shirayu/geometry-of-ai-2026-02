import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleMoE(nn.Module):
    """教育目的のMoE実装"""

    def __init__(self, d_model, num_experts=8, expert_capacity=2, expert_hidden=2048):
        super().__init__()
        self.num_experts = num_experts
        self.expert_capacity = expert_capacity

        # ルーター
        self.router = nn.Linear(d_model, num_experts)

        # Experts（簡単なFFN）
        self.experts = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(d_model, expert_hidden), nn.ReLU(), nn.Linear(expert_hidden, d_model))
                for _ in range(num_experts)
            ]
        )

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        batch_size, seq_len, d_model = x.shape

        # ルーティング
        router_logits = self.router(x)  # [batch, seq_len, num_experts]
        router_probs = F.softmax(router_logits, dim=-1)

        # Top-K選択
        topk_probs, topk_indices = torch.topk(router_probs, k=self.expert_capacity, dim=-1)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)  # 再正規化

        # Expert計算（簡略化: バッチ処理を省略）
        output = torch.zeros_like(x)
        for i in range(self.expert_capacity):
            expert_idx = topk_indices[..., i]  # [batch, seq_len]
            expert_weight = topk_probs[..., i].unsqueeze(-1)  # [batch, seq_len, 1]

            # 各Expertの出力を加重和（実装簡略化）
            for e in range(self.num_experts):
                mask = (expert_idx == e).unsqueeze(-1)  # [batch, seq_len, 1]
                expert_out = self.experts[e](x)
                output += expert_out * expert_weight * mask

        return output


# 使用例
moe = SimpleMoE(d_model=512, num_experts=8, expert_capacity=2)
x = torch.randn(2, 10, 512)  # [batch=2, seq_len=10, d_model=512]
out = moe(x)
print(f"Input shape: {x.shape}, Output shape: {out.shape}")
