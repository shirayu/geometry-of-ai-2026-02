import math

import torch
import torch.nn.functional as F


def sample_vmf(mu, kappa, num_samples=1):
    """vMF分布からサンプリング（Wood's algorithm）

    Args:
        mu: 平均方向 [batch, dim]
        kappa: 集中度 [batch, 1]
        num_samples: サンプル数

    Returns:
        samples: サンプル [batch, num_samples, dim]
    """
    batch_size, dim = mu.shape
    device = mu.device

    # κが非常に小さい場合は一様分布からサンプル
    if kappa.max() < 1e-6:
        samples = torch.randn(batch_size, num_samples, dim, device=device)
        return F.normalize(samples, dim=-1)

    # Householder変換でμを北極に写す
    # e1 = [1, 0, 0, ..., 0]
    e1 = torch.zeros(dim, device=device)
    e1[0] = 1.0

    results = []
    for _ in range(num_samples):
        # wをサンプル（μ方向の成分）
        w = _sample_w(kappa.squeeze(-1), dim)  # [batch]

        # 球面上の一様な方向をサンプル（μに直交する成分）
        v = torch.randn(batch_size, dim - 1, device=device)
        v = F.normalize(v, dim=-1)

        # 球面座標からデカルト座標へ
        sqrt_term = torch.sqrt(torch.clamp(1 - w**2, min=1e-10))

        # 北極周りのサンプルを構成
        sample_around_north = torch.zeros(batch_size, dim, device=device)
        sample_around_north[:, 0] = w
        sample_around_north[:, 1:] = sqrt_term.unsqueeze(-1) * v

        # Householder変換でμ周りに回転
        sample = _householder_rotation(sample_around_north, e1.expand(batch_size, -1), mu)
        results.append(sample)

    return torch.stack(results, dim=1)


def _sample_w(kappa, dim):
    """vMF分布のw成分をサンプル（rejection sampling）"""
    device = kappa.device
    batch_size = kappa.shape[0]

    # 近似パラメータ
    c = torch.sqrt(4 * kappa**2 + (dim - 1) ** 2)
    b = (c - 2 * kappa) / (dim - 1)
    a = ((dim - 1) + c) / (2 * kappa)

    # Rejection sampling
    w = torch.zeros(batch_size, device=device)
    done = torch.zeros(batch_size, dtype=torch.bool, device=device)

    max_iter = 1000
    for _ in range(max_iter):
        if done.all():
            break

        # 提案分布からサンプル
        eps = torch.rand(batch_size, device=device)
        u = torch.rand(batch_size, device=device)

        z = torch.cos(math.pi * eps)
        w_proposal = (1 + b * z) / (b + z)

        # 受理確率
        t = kappa * w_proposal + (dim - 1) * torch.log(1 - a * w_proposal)
        accept = (dim - 1) * torch.log(1 - a * b) - 1 + torch.log(u) < t

        w = torch.where(~done & accept, w_proposal, w)
        done = done | accept

    return w


def _householder_rotation(x, u, v):
    """Householder反射を用いてuをvに写す変換をxに適用

    uとvが単位ベクトルのとき、w = normalize(v - u) に対する
    Householder反射 H = I - 2ww^T は、uをvに写す。

    注意：厳密には反射であり回転ではない（行列式が-1）。
    vMFサンプリングの文脈では、北極周りの点をμ周りに移動させる
    という目的は多くの場合達成できる。ただし、以下の場合は
    2回の反射（＝回転）やRodriguesの回転公式を検討すること：
    - 時間方向の連続性・滑らかさが必要な場合
    - 後段処理が「回転」としての性質（det=+1）を仮定する場合
    """
    # u, vは単位ベクトル
    # 反射軸: w = normalize(v - u)
    w = v - u
    w_norm = w.norm(dim=-1, keepdim=True)

    # u ≈ v の場合（反射不要）
    mask = (w_norm < 1e-6).squeeze(-1)
    if mask.all().item():
        return x

    w = w / (w_norm + 1e-8)

    # Householder反射: x' = x - 2(x·w)w
    x_reflected = x - 2 * (x * w).sum(dim=-1, keepdim=True) * w

    # u ≈ v の場合は元のxを返す
    x_result = torch.where(mask.unsqueeze(-1), x, x_reflected)

    return x_result
