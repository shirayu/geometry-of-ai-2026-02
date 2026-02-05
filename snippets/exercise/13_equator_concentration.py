import numpy as np
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

for ax, n in zip(axes, [3, 100, 1000]):
    # n次元単位球面上の一様サンプル
    samples = np.random.randn(10000, n)
    samples = samples / np.linalg.norm(samples, axis=1, keepdims=True)

    x1 = samples[:, 0]  # 第1座標
    ax.hist(x1, bins=50, density=True)
    ax.set_title(f"n = {n}, std(x₁) = {np.std(x1):.4f}")
    ax.set_xlabel("x₁")

plt.tight_layout()
plt.show()
