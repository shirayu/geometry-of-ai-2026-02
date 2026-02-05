import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)


def check_orthogonality(dims, n_pairs=5000):
    """ランダムな単位ベクトル対の内積分布を計算"""
    results = []
    for d in dims:
        # 単位球面上の一様ランダムベクトル
        v1 = np.random.randn(n_pairs, d)
        v1 = v1 / np.linalg.norm(v1, axis=1, keepdims=True)
        v2 = np.random.randn(n_pairs, d)
        v2 = v2 / np.linalg.norm(v2, axis=1, keepdims=True)

        # 内積の計算
        dots = np.sum(v1 * v2, axis=1)

        # 角度に変換（度数法）
        angles = np.arccos(np.clip(dots, -1, 1)) * 180 / np.pi

        results.append(
            {
                "dim": d,
                "dot_mean": dots.mean(),
                "dot_std": dots.std(),
                "angle_mean": angles.mean(),
                "angle_std": angles.std(),
                "angles": angles,
            }
        )
    return results


dims = [3, 10, 100, 1000]
results = check_orthogonality(dims)

# 結果の表示
print("次元 | 内積平均 | 内積標準偏差 | 角度平均 | 角度標準偏差")
print("-" * 60)
for r in results:
    print(
        f"{r['dim']:4d} | {r['dot_mean']:+8.4f} | {r['dot_std']:12.4f} | "
        f"{r['angle_mean']:8.1f}° | {r['angle_std']:8.1f}°"
    )

# 角度のヒストグラム
fig, axes = plt.subplots(1, 4, figsize=(16, 3))
for ax, r in zip(axes, results, strict=False):
    ax.hist(r["angles"], bins=50, density=True, alpha=0.7)
    ax.axvline(90, color="red", linestyle="--", label="90°")
    ax.set_title(f"d = {r['dim']}")
    ax.set_xlabel("Angle (degrees)")
    ax.set_xlim(0, 180)
    ax.legend()
plt.tight_layout()
plt.show()
