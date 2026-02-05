import matplotlib.pyplot as plt
import numpy as np
import torch


class ResidualStack:
    def __init__(self, dim, num_blocks):
        self.dim = dim
        self.num_blocks = num_blocks

    def __call__(self, x, return_trajectory=False):
        trajectory = x.unsqueeze(0).repeat(self.num_blocks + 1, 1, 1)
        if return_trajectory:
            return x, trajectory
        return x


def visualize_trajectory_2d(trajectory, title="State Trajectory"):
    """2次元に射影した軌跡を可視化

    Args:
        trajectory: 軌跡 [time_steps, batch, dim]
        title: グラフのタイトル
    """
    # 最初の2次元に射影
    traj_2d = trajectory[:, :, :2].detach().cpu().numpy()
    time_steps, batch_size, _ = traj_2d.shape

    fig, ax = plt.subplots(figsize=(8, 8))

    # 各サンプルの軌跡をプロット
    colors = plt.cm.viridis(np.linspace(0, 1, batch_size))

    for i in range(min(batch_size, 20)):  # 最大20サンプル
        ax.plot(traj_2d[:, i, 0], traj_2d[:, i, 1], color=colors[i], alpha=0.5, linewidth=1)
        ax.scatter(
            traj_2d[0, i, 0],
            traj_2d[0, i, 1],
            color=colors[i],
            marker="o",
            s=50,
            label="Start" if i == 0 else "",
        )
        ax.scatter(
            traj_2d[-1, i, 0],
            traj_2d[-1, i, 1],
            color=colors[i],
            marker="x",
            s=50,
            label="End" if i == 0 else "",
        )

    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("trajectory_2d.png", dpi=150)
    plt.close()

    print("Saved: trajectory_2d.png")


# ResidualStackの軌跡を可視化
dim, num_blocks = 2, 20  # 2次元で直接可視化
model = ResidualStack(dim, num_blocks)

x = torch.randn(50, dim) * 2  # 50サンプル
output, trajectory = model(x, return_trajectory=True)

visualize_trajectory_2d(trajectory, "ResNet Trajectory (2D)")
