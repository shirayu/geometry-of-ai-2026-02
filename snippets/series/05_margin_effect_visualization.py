import matplotlib.pyplot as plt
import numpy as np


def visualize_margin_effect():
    """異なるマージン手法の効果を可視化"""
    theta = np.linspace(0, np.pi, 100)
    cos_theta = np.cos(theta)

    # 各手法のマージン適用後のコサイン値
    margin_arc = 0.5  # ArcFace: 約28.6度
    margin_cos = 0.35  # CosFace
    margin_sphere = 2  # SphereFace: m=2

    arcface = np.cos(theta + margin_arc)
    cosface = cos_theta - margin_cos
    sphereface = np.cos(margin_sphere * theta)

    plt.figure(figsize=(10, 6))
    plt.plot(np.degrees(theta), cos_theta, "k-", label="Original cos(θ)", linewidth=2)
    plt.plot(np.degrees(theta), arcface, "b-", label=f"ArcFace: cos(θ + {margin_arc})")
    plt.plot(np.degrees(theta), cosface, "r-", label=f"CosFace: cos(θ) - {margin_cos}")
    plt.plot(np.degrees(theta), sphereface, "g-", label=f"SphereFace: cos({margin_sphere}θ)")

    plt.xlabel("Angle θ (degrees)")
    plt.ylabel("Logit value")
    plt.title("Comparison of Angular Margin Methods")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 180)
    plt.ylim(-1.5, 1.5)
    plt.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    plt.show()


# 実行
visualize_margin_effect()
