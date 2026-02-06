import torch


def mobius_add(u, v, eps=1e-5):
    """メビウス加法（双曲空間でのベクトル加法）

    ポアンカレ円板モデルにおける「平行移動」に相当する操作
    """
    u_norm_sq = torch.sum(u * u, dim=-1, keepdim=True)
    v_norm_sq = torch.sum(v * v, dim=-1, keepdim=True)
    uv = torch.sum(u * v, dim=-1, keepdim=True)

    numerator = (1 + 2 * uv + v_norm_sq) * u + (1 - u_norm_sq) * v
    denominator = 1 + 2 * uv + u_norm_sq * v_norm_sq + eps

    return numerator / denominator


def exp_map_0(v, eps=1e-5):
    """原点における指数写像

    接空間（ユークリッド）のベクトルを、
    ポアンカレ円板上の点に写す

    Args:
        v: 接空間のベクトル [batch, dim]

    Returns:
        ポアンカレ円板上の点
    """
    v_norm = torch.clamp(torch.norm(v, dim=-1, keepdim=True), min=eps)
    return torch.tanh(v_norm) * v / v_norm


def log_map_0(y, eps=1e-5):
    """原点における対数写像

    ポアンカレ円板上の点を、
    接空間（ユークリッド）のベクトルに写す

    Args:
        y: ポアンカレ円板上の点 [batch, dim]

    Returns:
        接空間のベクトル
    """
    y_norm = torch.clamp(torch.norm(y, dim=-1, keepdim=True), min=eps, max=1 - eps)
    return torch.atanh(y_norm) * y / y_norm


def project_to_poincare(x, eps=1e-5):
    """点をポアンカレ円板内に射影

    境界を超えた点を、境界内に引き戻す
    """
    norm = torch.norm(x, dim=-1, keepdim=True)
    max_norm = 1 - eps
    cond = norm > max_norm
    return torch.where(cond, x / norm * max_norm, x)


# 使用例：リーマン勾配降下法の1ステップ
def riemannian_sgd_step(x, euclidean_grad, lr=0.01):
    """リーマン勾配降下法の1ステップ

    1. ユークリッド勾配をリーマン勾配に変換
    2. 接空間で更新
    3. 多様体に射影
    """
    # リーマン計量によるスケーリング
    # ポアンカレ円板の計量テンソル: g_x = (2 / (1 - ||x||^2))^2 * I
    x_norm_sq = torch.sum(x * x, dim=-1, keepdim=True)
    scale = ((1 - x_norm_sq) ** 2) / 4
    riemannian_grad = scale * euclidean_grad

    # 接空間での更新（対数写像で接空間へ、更新後、指数写像で戻る）
    # 簡易版：直接更新して射影
    x_new = x - lr * riemannian_grad
    x_new = project_to_poincare(x_new)

    return x_new


# テスト
x = torch.tensor([[0.3, 0.4]])
grad = torch.tensor([[0.1, 0.1]])
x_new = riemannian_sgd_step(x, grad, lr=0.1)
print(f"Before: {x}, norm={torch.norm(x):.4f}")
print(f"After: {x_new}, norm={torch.norm(x_new):.4f}")
