import torch
import torch.nn.functional as F

# 正規化前
x = torch.tensor([[3.0, 4.0], [1.0, 2.0]])
y = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

dot_before = torch.sum(x * y, dim=-1)
print(f"正規化前の内積: {dot_before}")  # [3.0, 2.0]

# 正規化後
x_norm = F.normalize(x, p=2, dim=-1)
y_norm = F.normalize(y, p=2, dim=-1)

dot_after = torch.sum(x_norm * y_norm, dim=-1)
print(f"正規化後の内積: {dot_after}")  # [0.6, 0.894...]

# 確認：正規化後の内積はcosθに一致
print(f"||x_norm||: {torch.norm(x_norm, dim=-1)}")  # [1.0, 1.0]
