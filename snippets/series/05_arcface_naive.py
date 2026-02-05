import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ArcFaceNaive(nn.Module):
    """ArcFaceの素朴な実装（教育目的）
    
    注意: この実装は数値的に不安定な場合がある。
    実用には後述の安定版を使用すること。
    """
    
    def __init__(self, in_features, num_classes, scale=30.0, margin=0.5):
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.randn(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, x, labels):
        # 特徴と重みを正規化
        x_norm = F.normalize(x, dim=1)
        w_norm = F.normalize(self.weight, dim=1)
        
        # コサイン類似度
        cosine = F.linear(x_norm, w_norm)  # [batch, num_classes]
        
        # 正解クラスの角度にマージンを加算
        # acos → +m → cos という経路
        theta = torch.acos(torch.clamp(cosine, -1 + 1e-7, 1 - 1e-7))
        target_logits = torch.cos(theta[range(len(labels)), labels] + self.margin)
        
        # 正解クラスのロジットを置き換え
        logits = cosine.clone()
        logits[range(len(labels)), labels] = target_logits
        
        # スケーリング
        return logits * self.scale
