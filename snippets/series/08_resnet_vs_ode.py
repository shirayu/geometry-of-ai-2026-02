import torch

TORCHDIFFEQ_AVAILABLE = False


class ResidualStack:
    def __init__(self, dim, num_steps):
        self.dim = dim
        self.num_steps = num_steps

    def __call__(self, x, return_trajectory=False):
        trajectory = x.unsqueeze(0).repeat(self.num_steps + 1, 1, 1)
        if return_trajectory:
            return x, trajectory
        return x


class NeuralODE:
    def __init__(self, dim):
        self.dim = dim

    def __call__(self, x, t_span, return_trajectory=False):
        trajectory = x.unsqueeze(0).repeat(t_span.numel(), 1, 1)
        if return_trajectory:
            return x, trajectory
        return x


def compare_resnet_and_ode(dim=64, num_steps=10):
    """ResNetとNeural ODEの挙動を比較

    ResNet: 離散的なステップ
    Neural ODE: 連続的な流れ（離散化して比較）
    """
    # ResNet
    resnet = ResidualStack(dim, num_steps)

    # Neural ODE（利用可能な場合）
    if TORCHDIFFEQ_AVAILABLE:
        neural_ode = NeuralODE(dim)

    # テスト入力
    x = torch.randn(16, dim)

    # ResNetの軌跡
    _, resnet_traj = resnet(x, return_trajectory=True)

    results = {
        "resnet": {
            "trajectory_shape": resnet_traj.shape,
            "output_norm_mean": resnet_traj[-1].norm(dim=-1).mean().item(),
            "output_norm_std": resnet_traj[-1].norm(dim=-1).std().item(),
        }
    }

    # Neural ODEの軌跡（利用可能な場合）
    if TORCHDIFFEQ_AVAILABLE:
        t_span = torch.linspace(0, 1, num_steps + 1)
        _, ode_traj = neural_ode(x, t_span, return_trajectory=True)

        results["neural_ode"] = {
            "trajectory_shape": ode_traj.shape,
            "output_norm_mean": ode_traj[-1].norm(dim=-1).mean().item(),
            "output_norm_std": ode_traj[-1].norm(dim=-1).std().item(),
        }

    return results


# 比較実行
results = compare_resnet_and_ode()
print("Comparison Results:")
for model_name, metrics in results.items():
    print(f"\n{model_name}:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
