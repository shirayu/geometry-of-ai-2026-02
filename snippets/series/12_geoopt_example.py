# pip install geoopt

import geoopt
import torch


def geoopt_poincare_example():
    """Geooptライブラリを使ったポアンカレ球での操作例"""

    # ポアンカレ球多様体（デフォルトは曲率-1）
    ball = geoopt.PoincareBall()

    # ランダムな点を生成（多様体上に）
    x = ball.random(2, 5)  # [2, 5] の形状
    print(f"Points on Poincaré ball:\n{x}")
    print(f"Norms: {torch.norm(x, dim=-1)}")  # 必ず < 1

    # 2点間の距離
    y = ball.random(2, 5)
    dist = ball.dist(x, y)
    print(f"Distances: {dist}")

    # 指数写像（接空間 → 多様体）
    origin = torch.zeros(5)
    v = torch.randn(5) * 0.1  # 接ベクトル
    point = ball.expmap(origin, v)
    print(f"Exp map result: {point}, norm={torch.norm(point):.4f}")

    # 対数写像（多様体 → 接空間）
    v_back = ball.logmap(origin, point)
    print(f"Log map result: {v_back}")
    print(f"Original v: {v}")

    # 測地線に沿った補間
    t = 0.5
    midpoint = ball.geodesic(t, x[0], y[0])
    print(f"Midpoint on geodesic: {midpoint}")


def geoopt_optimization_example():
    """Geooptを使ったリーマン最適化の例"""

    ball = geoopt.PoincareBall()

    # 多様体上のパラメータ
    x = geoopt.ManifoldParameter(ball.random(10, 5), manifold=ball)

    # リーマン最適化器
    optimizer = geoopt.optim.RiemannianAdam([x], lr=0.01)

    # ダミーの最適化（中心に向かう）
    for i in range(100):
        optimizer.zero_grad()
        loss = torch.sum(x**2)  # 中心（原点）に近づける
        loss.backward()
        optimizer.step()

        if (i + 1) % 20 == 0:
            avg_norm = torch.norm(x, dim=-1).mean()
            print(f"Step {i + 1}, Loss: {loss.item():.4f}, Avg norm: {avg_norm:.4f}")


if __name__ == "__main__":
    print("=== Poincaré Ball Operations ===")
    geoopt_poincare_example()
    print("\n=== Riemannian Optimization ===")
    geoopt_optimization_example()
