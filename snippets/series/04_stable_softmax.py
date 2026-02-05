import torch
import torch.nn.functional as F

def stable_softmax(logits, temperature=1.0, dim=-1):
    """数値安定なSoftmax
    
    Args:
        logits: 入力ロジット
        temperature: 温度パラメータ（default: 1.0）
        dim: softmaxを適用する次元
    
    Returns:
        確率分布
    """
    # 温度でスケーリング
    scaled = logits / temperature
    # 最大値を引いてオーバーフローを防ぐ
    # （PyTorchのsoftmaxは内部でこれを行うが、明示的に示す）
    shifted = scaled - scaled.max(dim=dim, keepdim=True).values
    exp_shifted = torch.exp(shifted)
    return exp_shifted / exp_shifted.sum(dim=dim, keepdim=True)

# PyTorchの組み込み関数を使う場合
def softmax_with_temperature(logits, temperature=1.0, dim=-1):
    """温度付きSoftmax（推奨）"""
    return F.softmax(logits / temperature, dim=dim)

# 使用例
logits = torch.tensor([2.0, 1.0, 0.1])

print("τ=0.5 (低温):", softmax_with_temperature(logits, 0.5))
print("τ=1.0 (標準):", softmax_with_temperature(logits, 1.0))
print("τ=2.0 (高温):", softmax_with_temperature(logits, 2.0))

# 期待される結果：
# τ=0.5: [0.88, 0.12, 0.01] に近い（最大に集中）
# τ=1.0: [0.66, 0.24, 0.10] に近い
# τ=2.0: [0.49, 0.32, 0.19] に近い（一様に近づく）
