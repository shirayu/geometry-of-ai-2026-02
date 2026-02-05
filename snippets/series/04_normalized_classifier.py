import torch
import torch.nn as nn
import torch.nn.functional as F


class NormalizedClassifier(nn.Module):
    """正規化された分類器（角度ベースの分類）

    入力と重みの両方を正規化し、ロジットを純粋な角度情報にする。
    """

    def __init__(self, in_features, num_classes, scale=30.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, in_features))
        self.scale = scale  # ロジットのスケール（温度の逆数的な役割）

        # 重みを正規化して初期化
        with torch.no_grad():
            self.weight.data = F.normalize(self.weight.data, dim=1)

    def forward(self, x):
        # 入力を正規化
        x_norm = F.normalize(x, dim=-1)
        # 重みを正規化
        w_norm = F.normalize(self.weight, dim=1)
        # 内積 = cosine similarity
        logits = F.linear(x_norm, w_norm)
        # スケーリング（角度マージンの効果を出すため）
        return logits * self.scale


# 使用例
classifier = NormalizedClassifier(768, 10)
embeddings = torch.randn(32, 768)
logits = classifier(embeddings)

# ロジットの範囲を確認
print(f"Logits range: [{logits.min():.2f}, {logits.max():.2f}]")
# 期待値: [-scale, +scale] の範囲（cosineは[-1, 1]なので）
