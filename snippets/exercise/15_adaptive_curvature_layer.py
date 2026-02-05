class AdaptiveCurvatureLayer(nn.Module):
    def __init__(self, dim):
        self.kappa = nn.Parameter(torch.zeros(1))  # 学習可能な曲率
        
    def forward(self, x):
        if self.kappa > threshold:
            return spherical_projection(x, self.kappa)
        elif self.kappa < -threshold:
            return hyperbolic_projection(x, -self.kappa)
        else:
            return x  # ユークリッドのまま
