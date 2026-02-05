import torch


def slerp(v0, v1, t):
    """球面線形補間（Slerp）

    Args:
        v0, v1: 単位ベクトル（正規化済み）
        t: 補間パラメータ（0から1）

    Returns:
        球面上の測地線に沿った補間点
    """
    # 内積からなす角を計算
    dot = torch.clamp(torch.sum(v0 * v1), -1.0, 1.0)
    theta = torch.acos(dot)

    # 特殊ケース：ほぼ同じ方向
    if theta.abs() < 1e-6:
        return v0

    # Slerp公式
    sin_theta = torch.sin(theta)
    return (torch.sin((1 - t) * theta) / sin_theta) * v0 + (
        torch.sin(t * theta) / sin_theta
    ) * v1
