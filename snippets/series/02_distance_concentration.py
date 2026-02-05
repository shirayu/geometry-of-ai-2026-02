import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)


def distance_concentration(dims, n_samples=10000):
    """各次元での原点からの距離分布を計算"""
    results = []
    for d in dims:
        # 標準正規分布からサンプリング
        X = np.random.randn(n_samples, d)
        distances = np.linalg.norm(X, axis=1)
        results.append(
            {
                "dim": d,
                "mean": distances.mean(),
                "std": distances.std(),
                "relative_std": distances.std() / distances.mean(),
                "distances": distances,
            }
        )
    return results


dims = [3, 10, 100, 1000]
results = distance_concentration(dims)

# 結果の表示
print("次元 | 平均距離 | 標準偏差 | 相対標準偏差")
print("-" * 45)
for r in results:
    print(
        f"{r['dim']:4d} | {r['mean']:8.2f} | {r['std']:8.2f} | {r['relative_std']:8.4f}"
    )

# ヒストグラムの可視化
fig, axes = plt.subplots(1, 4, figsize=(16, 3))
for ax, r in zip(axes, results):
    ax.hist(r["distances"], bins=50, density=True, alpha=0.7)
    ax.axvline(r["mean"], color="red", linestyle="--", label=f"mean={r['mean']:.1f}")
    ax.set_title(f"d = {r['dim']}")
    ax.set_xlabel("Distance from origin")
    ax.legend()
plt.tight_layout()
plt.show()
