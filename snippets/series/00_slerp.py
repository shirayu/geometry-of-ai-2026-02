import torch


def slerp(v0, v1, t):
    """球面線形補間"""
    dot = torch.clamp(torch.sum(v0 * v1), -1.0, 1.0)
    theta = torch.acos(dot)
    if theta.abs() < 1e-6:
        return v0
    sin_theta = torch.sin(theta)
    return (torch.sin((1 - t) * theta) / sin_theta) * v0 + (
        torch.sin(t * theta) / sin_theta
    ) * v1


def lerp(v0, v1, t):
    """線形補間（正規化なし）"""
    return (1 - t) * v0 + t * v1


def lerp_normalized(v0, v1, t):
    """線形補間して正規化"""
    v = (1 - t) * v0 + t * v1
    return v / v.norm()


# 3次元で可視化
v0 = torch.tensor([1.0, 0.0, 0.0])
v1 = torch.tensor([0.0, 1.0, 0.0])
ts = torch.linspace(0, 1, 20)

slerp_points = torch.stack([slerp(v0, v1, t) for t in ts])
lerp_points = torch.stack([lerp(v0, v1, t) for t in ts])
lerp_norm_points = torch.stack([lerp_normalized(v0, v1, t) for t in ts])

# 結果：
# - slerp_points: 球面上の大円に沿った点列
# - lerp_points: 球面を突き抜ける直線（中点でノルム≈0.71）
# - lerp_norm_points: slerpとほぼ同じ（この場合）
