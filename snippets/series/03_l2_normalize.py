import torch
import torch.nn.functional as F


def l2_normalize(x, dim=-1, eps=1e-12):
    """L2正規化（球面への射影）

    Args:
        x: 入力テンソル
        dim: 正規化する次元
        eps: 数値安定性のための小さな値

    Returns:
        単位ノルムに正規化されたテンソル
    """
    return F.normalize(x, p=2, dim=dim, eps=eps)


# 使用例
embeddings = torch.randn(32, 768)  # バッチサイズ32、次元768
normalized = l2_normalize(embeddings)

# 確認：ノルムが1になっている
print(f"ノルムの平均: {normalized.norm(dim=-1).mean():.6f}")  # ≈ 1.0
print(f"ノルムの標準偏差: {normalized.norm(dim=-1).std():.6f}")  # ≈ 0.0
