import numpy as np
from ripser import ripser


def compute_persistence(data, maxdim=1):
    """点群のパーシステントホモロジーを計算

    Args:
        data: 点群 (n_points, n_features)
        maxdim: 計算する最大次元（0=連結成分, 1=ループ, 2=空洞, ...）

    Returns:
        パーシステンス図のリスト（各次元に対して1つ）
    """
    result = ripser(data, maxdim=maxdim)
    return result["dgms"]


# 例1: ノイズ付きの円
np.random.seed(42)
n_points = 100
theta = 2 * np.pi * np.random.rand(n_points)
noise = 0.1 * np.random.randn(n_points, 2)
circle = np.column_stack([np.cos(theta), np.sin(theta)]) + noise

# パーシステンス図の計算
diagrams = compute_persistence(circle, maxdim=1)

print("H0 (連結成分):", diagrams[0].shape[0], "個の特徴")
print("H1 (ループ):", diagrams[1].shape[0], "個の特徴")

# 最も持続性の高いH1特徴（円の「穴」に対応）
h1_persistence = diagrams[1][:, 1] - diagrams[1][:, 0]
print(f"最大持続性: {h1_persistence.max():.3f}")
