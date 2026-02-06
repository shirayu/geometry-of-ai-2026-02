import torch


def poincare_distance(u, v, eps=1e-5):
    """ポアンカレ円板モデルにおける双曲距離

    Args:
        u: 点1 [batch, dim] または [dim]
        v: 点2 [batch, dim] または [dim]
        eps: 数値安定性のための小さな値

    Returns:
        双曲距離
    """
    # ノルムの2乗
    u_norm_sq = torch.clamp(torch.sum(u * u, dim=-1), max=1 - eps)
    v_norm_sq = torch.clamp(torch.sum(v * v, dim=-1), max=1 - eps)
    diff_norm_sq = torch.sum((u - v) ** 2, dim=-1)

    # 距離公式
    numerator = 2 * diff_norm_sq
    denominator = (1 - u_norm_sq) * (1 - v_norm_sq)

    # arcosh(1 + x) = log(1 + x + sqrt(x^2 + 2x)) for numerical stability
    x = numerator / (denominator + eps)
    return torch.log(1 + x + torch.sqrt(x * x + 2 * x + eps))


# 使用例
u = torch.tensor([0.0, 0.0])  # 中心
v = torch.tensor([0.5, 0.0])  # 中心から離れた点
w = torch.tensor([0.9, 0.0])  # 境界に近い点

print(f"d(center, mid): {poincare_distance(u, v):.4f}")
print(f"d(center, edge): {poincare_distance(u, w):.4f}")
print(f"d(mid, edge): {poincare_distance(v, w):.4f}")

# 期待される結果：
# 境界に近い点への距離が非常に大きくなる
