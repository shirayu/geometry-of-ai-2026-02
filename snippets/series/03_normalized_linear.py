import torch.nn as nn

class NormalizedLinear(nn.Module):
    """正規化を含む線形層（nGPT風）"""
    
    def __init__(self, in_features, out_features, eps=1e-12):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.eps = eps
        
        # 重みも正規化して初期化
        with torch.no_grad():
            self.linear.weight.data = F.normalize(
                self.linear.weight.data, dim=1, eps=eps
            )
    
    def forward(self, x):
        # 入力は正規化済みと仮定
        out = self.linear(x)
        # 出力を正規化
        return F.normalize(out, dim=-1, eps=self.eps)

class NormalizedResidual(nn.Module):
    """正規化を含む残差接続（nGPT風）"""
    
    def __init__(self, module, alpha=0.1, eps=1e-12):
        super().__init__()
        self.module = module
        self.alpha = alpha  # 残差のスケール
        self.eps = eps
    
    def forward(self, x):
        # 残差を加算
        out = x + self.alpha * self.module(x)
        # 正規化して球面に戻す
        return F.normalize(out, dim=-1, eps=self.eps)
