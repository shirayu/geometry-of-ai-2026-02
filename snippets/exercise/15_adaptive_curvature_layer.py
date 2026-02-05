import torch
import torch.nn as nn


def spherical_projection(x, kappa):
    return x


def hyperbolic_projection(x, kappa):
    return x


class AdaptiveCurvatureLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.kappa = nn.Parameter(torch.zeros(1))  # 学習可能な曲率

    def forward(self, x):
        threshold = 1e-3
        if self.kappa > threshold:
            return spherical_projection(x, self.kappa)
        elif self.kappa < -threshold:
            return hyperbolic_projection(x, -self.kappa)
        else:
            return x  # ユークリッドのまま
