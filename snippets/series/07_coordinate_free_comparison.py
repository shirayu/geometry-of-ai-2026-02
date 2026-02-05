import torch


def compare_distributions_coordinate_free(kappa1, kappa2):
    """座標系に依存しない分布の比較

    確信度（κ）のみを使った比較。
    座標系の整列なしで、2つのモデルの「確信度」を比較できる。

    Args:
        kappa1, kappa2: 2つのモデルからのκ [batch, 1]

    Returns:
        comparison: 比較結果の辞書
    """
    kappa1 = kappa1.squeeze(-1)
    kappa2 = kappa2.squeeze(-1)

    return {
        "kappa1_mean": kappa1.mean().item(),
        "kappa2_mean": kappa2.mean().item(),
        "kappa_diff": (kappa1 - kappa2).mean().item(),
        "kappa_ratio": (kappa1 / (kappa2 + 1e-8)).mean().item(),
        "more_confident": "model1" if kappa1.mean() > kappa2.mean() else "model2",
        "correlation": torch.corrcoef(torch.stack([kappa1, kappa2]))[0, 1].item(),
    }


# 使用例：2つのモデルの確信度を比較
kappa_model_a = torch.tensor([50.0, 30.0, 45.0, 20.0])
kappa_model_b = torch.tensor([40.0, 35.0, 50.0, 15.0])

comparison = compare_distributions_coordinate_free(
    kappa_model_a.unsqueeze(-1), kappa_model_b.unsqueeze(-1)
)

print("Coordinate-free comparison:")
for k, v in comparison.items():
    print(f"  {k}: {v}")
