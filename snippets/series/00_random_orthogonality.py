import torch


def check_random_orthogonality(dims, n_pairs=1000):
    """高次元でランダムベクトルの内積分布を調べる"""
    results = []

    for d in dims:
        # 単位球面上の一様ランダムベクトル
        v1 = torch.randn(n_pairs, d)
        v1 = v1 / v1.norm(dim=1, keepdim=True)
        v2 = torch.randn(n_pairs, d)
        v2 = v2 / v2.norm(dim=1, keepdim=True)

        # 内積の計算
        dots = (v1 * v2).sum(dim=1)
        results.append({"dim": d, "mean": dots.mean().item(), "std": dots.std().item()})

    return results


# 実験
dims = [10, 100, 1000, 10000]
results = check_random_orthogonality(dims)

for r in results:
    print(f"d={r['dim']:5d}: mean={r['mean']:+.4f}, std={r['std']:.4f}")

# 期待される結果：
# d=   10: mean≈0, std≈0.32
# d=  100: mean≈0, std≈0.10
# d= 1000: mean≈0, std≈0.032
# d=10000: mean≈0, std≈0.010
# → 次元が上がると標準偏差が 1/√d で減少
