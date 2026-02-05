class ArcFaceStable(nn.Module):
    """数値安定なArcFace実装
    
    cos(θ + m) = cos(θ)cos(m) - sin(θ)sin(m) を利用し、
    acos を避けることで数値安定性を確保する。
    """
    
    def __init__(self, in_features, num_classes, scale=30.0, margin=0.5):
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.randn(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
        
        # マージンの三角関数を事前計算
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        # cos(π - m) を使った閾値（θ + m > π を防ぐ）
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
    
    def forward(self, x, labels):
        # 特徴と重みを正規化
        x_norm = F.normalize(x, dim=1)
        w_norm = F.normalize(self.weight, dim=1)
        
        # コサイン類似度
        cosine = F.linear(x_norm, w_norm)
        
        # sin(θ) = sqrt(1 - cos²(θ))
        sine = torch.sqrt(torch.clamp(1.0 - cosine ** 2, min=1e-9))
        
        # cos(θ + m) = cos(θ)cos(m) - sin(θ)sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m
        
        # θ + m > π の場合の処理（角度が大きすぎる場合）
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        # 正解クラスのみマージンを適用
        one_hot = F.one_hot(labels, num_classes=self.weight.size(0)).float()
        logits = one_hot * phi + (1.0 - one_hot) * cosine
        
        return logits * self.scale

# 使用例
def train_step(model, arcface, x, labels, optimizer, criterion):
    """ArcFaceを使った学習ステップ"""
    optimizer.zero_grad()
    
    # 特徴抽出
    features = model(x)  # [batch, feature_dim]
    
    # ArcFaceでロジット計算
    logits = arcface(features, labels)  # [batch, num_classes]
    
    # クロスエントロピー損失
    loss = criterion(logits, labels)
    
    loss.backward()
    optimizer.step()
    
    return loss.item()
