import matplotlib.pyplot as plt
import numpy as np
from persim import plot_diagrams
from ripser import ripser


def visualize_persistence(data, title="Persistence Diagram"):
    """点群とパーシステンス図を並べて表示"""
    result = ripser(data, maxdim=1)
    diagrams = result["dgms"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 点群の表示
    axes[0].scatter(data[:, 0], data[:, 1], s=20, alpha=0.7)
    axes[0].set_aspect("equal")
    axes[0].set_title("Point Cloud")

    # パーシステンス図の表示
    plot_diagrams(diagrams, ax=axes[1])
    axes[1].set_title(title)

    plt.tight_layout()
    return fig


# 例: 異なる形状の比較
np.random.seed(42)
n = 100

# 円
theta = 2 * np.pi * np.random.rand(n)
circle = np.column_stack([np.cos(theta), np.sin(theta)]) + 0.1 * np.random.randn(n, 2)

# 2つの円（8の字に近い）
theta1 = 2 * np.pi * np.random.rand(n // 2)
theta2 = 2 * np.pi * np.random.rand(n // 2)
two_circles = np.vstack(
    [
        np.column_stack([np.cos(theta1) - 1, np.sin(theta1)]),
        np.column_stack([np.cos(theta2) + 1, np.sin(theta2)]),
    ]
)
two_circles += 0.1 * np.random.randn(n, 2)

# ランダムな点群
random_cloud = np.random.randn(n, 2)

# 可視化（実行時）
# visualize_persistence(circle, "Circle: 1 loop expected")
# visualize_persistence(two_circles, "Two circles: 2 loops expected")
# visualize_persistence(random_cloud, "Random: no persistent loops expected")
