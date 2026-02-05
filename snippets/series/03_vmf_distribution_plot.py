import matplotlib.pyplot as plt
import numpy as np
from scipy.special import iv  # ベッセル関数


def vmf_pdf_2d(theta, kappa):
    """2次元（円上）のvMF分布の確率密度

    平均方向を θ=0 として、角度 θ での密度を返す
    """
    # 2次元の場合、正規化定数は 1/(2π I_0(κ))
    normalization = 1 / (2 * np.pi * iv(0, kappa))
    return normalization * np.exp(kappa * np.cos(theta))


# 異なる κ での分布を可視化
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左：確率密度のプロット
theta = np.linspace(-np.pi, np.pi, 1000)
kappas = [0, 1, 5, 10, 50]

ax = axes[0]
for kappa in kappas:
    if kappa == 0:
        # κ=0 は一様分布
        pdf = np.ones_like(theta) / (2 * np.pi)
    else:
        pdf = vmf_pdf_2d(theta, kappa)
    ax.plot(theta, pdf, label=f"κ = {kappa}")

ax.set_xlabel("Angle θ (radians)")
ax.set_ylabel("Probability Density")
ax.set_title("von Mises-Fisher Distribution (2D)")
ax.legend()
ax.set_xlim(-np.pi, np.pi)

# 右：極座標でのプロット（円上の分布）
ax = axes[1]
for kappa in kappas:
    if kappa == 0:
        r = np.ones_like(theta) / (2 * np.pi)
    else:
        r = vmf_pdf_2d(theta, kappa)
    # 極座標での x, y
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    ax.plot(x, y, label=f"κ = {kappa}")

ax.set_aspect("equal")
ax.set_title("vMF Distribution on Circle")
ax.legend()

plt.tight_layout()
plt.show()
