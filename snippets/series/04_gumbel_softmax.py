import torch
import torch.nn.functional as F

def gumbel_softmax(logits, temperature=1.0, hard=False):
    """Gumbel-Softmax（微分可能な離散サンプリング）
    
    Args:
        logits: 入力ロジット [batch, num_classes]
        temperature: 温度パラメータ
        hard: Trueなら順伝播でone-hot、逆伝播でソフト勾配（STE）
    
    Returns:
        サンプリングされた確率ベクトル
    """
    # PyTorchの組み込み関数を使用
    return F.gumbel_softmax(logits, tau=temperature, hard=hard)

# 使用例
logits = torch.tensor([[2.0, 1.0, 0.1]])

# ソフトサンプリング（微分可能）
soft_sample = gumbel_softmax(logits, temperature=0.5, hard=False)
print("Soft sample:", soft_sample)

# ハードサンプリング（STE）
hard_sample = gumbel_softmax(logits, temperature=0.5, hard=True)
print("Hard sample:", hard_sample)  # one-hotに近い

# 温度による変化を可視化
temperatures = [0.1, 0.5, 1.0, 2.0, 5.0]
for tau in temperatures:
    samples = torch.stack([gumbel_softmax(logits, tau, hard=False) 
                          for _ in range(1000)])
    print(f"τ={tau}: mean={samples.mean(dim=0)}, "
          f"std={samples.std(dim=0)}")
