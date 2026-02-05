import math

import torch


def kl_divergence_vmf_approx(mu1, kappa1, mu2, kappa2):
    """2つのvMF分布間のKLダイバージェンス【高κ近似版】

    KL(vMF(μ1, κ1) || vMF(μ2, κ2))

    WARNING: この実装は高κ・高次元での漸近近似を使用。
    κ < 10 程度の領域では誤差が大きくなる可能性がある。

    Args:
        mu1, mu2: 平均方向 [batch, dim]
        kappa1, kappa2: 集中度 [batch, 1]

    Returns:
        kl: KLダイバージェンス [batch]（近似値）
    """
    dim = mu1.shape[-1]

    # コサイン類似度
    cos_sim = (mu1 * mu2).sum(dim=-1, keepdim=True)

    # Bessel関数の比（高κ近似）
    A1 = _bessel_ratio_approx(kappa1, dim)

    # 正規化定数の差（対数、高κ近似）
    log_C1 = _log_normalizer_vmf_approx(kappa1, dim)
    log_C2 = _log_normalizer_vmf_approx(kappa2, dim)

    # KLダイバージェンス
    kl = log_C2 - log_C1 + kappa1 * A1 - kappa2 * A1 * cos_sim

    return kl.squeeze(-1)


def _bessel_ratio_approx(kappa, dim):
    """Bessel関数の比 I_{d/2}(κ) / I_{d/2-1}(κ) の高κ近似

    WARNING: κが小さい（< 10程度）場合、この近似は不正確。
    厳密計算には scipy.special.ive を使用すること。
    """
    # 高κでの近似: A(κ) ≈ 1 - (d-1)/(2κ)
    nu = dim / 2 - 1
    return 1 - (2 * nu + 1) / (2 * kappa + 1e-8)


def _log_normalizer_vmf_approx(kappa, dim):
    """vMF分布の対数正規化定数 log C_d(κ) の高κ近似

    WARNING: κが小さい（< 10程度）場合、この近似は不正確。
    厳密計算には scipy.special.ive を使用すること。

    近似式: log C ≈ (d/2 - 1) * log κ - d/2 * log(2π) - κ
    （Stirling近似に基づく、高κ・高次元での漸近展開）
    """
    nu = dim / 2 - 1
    return nu * torch.log(kappa + 1e-8) - (dim / 2) * math.log(2 * math.pi) - kappa


# 参考：厳密計算が必要な場合のスケルトン
def _log_normalizer_vmf_exact(kappa, dim):
    """vMF分布の対数正規化定数（厳密版、要scipy）

    log C_d(κ) = (d/2 - 1) * log κ - (d/2) * log(2π) - log I_{d/2-1}(κ)

    数値安定性のため、スケール付きBessel関数 ive を使用：
    I_ν(κ) = ive(ν, κ) * exp(κ)
    よって log I_ν(κ) = log(ive(ν, κ)) + κ

    WARNING: この関数はNumPy配列またはPython floatを想定。
    PyTorch tensorを渡す場合は事前に変換すること：
        kappa_np = kappa.detach().cpu().numpy()
        result_np = _log_normalizer_vmf_exact(kappa_np, dim)
        result = torch.from_numpy(result_np).to(kappa.device)
    """
    # from scipy.special import ive
    # import numpy as np
    # nu = dim / 2 - 1
    # # ive(nu, kappa) は exp(-kappa) * I_nu(kappa) を返す（オーバーフロー防止）
    # log_ive = np.log(np.maximum(ive(nu, kappa), 1e-300))
    # log_bessel = log_ive + kappa  # log I_nu(kappa) を復元
    # return nu * np.log(kappa) - (dim / 2) * np.log(2 * np.pi) - log_bessel
    raise NotImplementedError("厳密計算にはscipyが必要。上記コメントを参照。")


# 互換性のためのエイリアス（旧名で呼び出すコードがある場合用）
kl_divergence_vmf = kl_divergence_vmf_approx
