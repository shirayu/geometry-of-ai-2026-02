import torch
import torch.nn.functional as F


def demonstrate_instability():
    """float16での数値不安定性の例"""

    # float16では非常に小さい値が0に丸められやすい（アンダーフロー）
    # min positive normal ≈ 6e-5 なので、それより十分小さい値は 0 になりやすい
    very_small = torch.finfo(torch.float16).tiny / 1024  # float32では非ゼロ
    small_fp16 = torch.full(
        (3,), very_small, dtype=torch.float32
    ).half()  # → 0 になりうる

    print(f"入力 (float16): {small_fp16}")
    print(f"ノルム: {small_fp16.norm()}")  # 非常に小さい、または0

    # eps なしで正規化すると不安定（ノルムが 0 に丸められると NaN/Inf）
    try:
        normalized_unsafe = small_fp16 / small_fp16.norm()
        print(f"Unsafe (epsなし): {normalized_unsafe}")
    except Exception as e:
        print(f"Unsafe normalization failed: {e}")

    # 実務では「ゼロベクトルが混ざる」ケースも多い（マスクや初期化など）
    zero_fp16 = torch.zeros(3, dtype=torch.float16)
    print(f"ゼロ入力 (float16): {zero_fp16}, ノルム: {zero_fp16.norm()}")
    print(f"ゼロをepsなしで割る: {zero_fp16 / zero_fp16.norm()}")  # NaN が出る

    # 安全策：ノルム計算をfloat32で行い、epsで保護して正規化
    normalized_safe = F.normalize(small_fp16.float(), eps=1e-12).half()
    print(f"Safe (eps使用, normはfloat32): {normalized_safe}")

    # 勾配計算時にゼロ近傍で問題が起きる例
    x = torch.tensor([1e-7, 1e-7, 1e-7], requires_grad=True)
    y = x / (x.norm() + 1e-12)  # eps で保護
    y.sum().backward()
    print(f"勾配 (eps保護あり): {x.grad}")


demonstrate_instability()
