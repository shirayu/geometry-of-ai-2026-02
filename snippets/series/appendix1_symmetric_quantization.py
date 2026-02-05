import torch

def quantize_tensor_symmetric(x, bits=8, eps=1e-8):
    """対称量子化の概念例（スカラーscale）。
    実務では per-channel / group-wise がよく使われる。
    """
    qmin, qmax = -(2**(bits-1)), 2**(bits-1) - 1
    scale = x.abs().max().clamp_min(eps) / qmax
    x_q = torch.round(x / scale).clamp(qmin, qmax)
    return x_q, scale

def dequantize_tensor(x_q, scale):
    return x_q * scale

# 使用例
weight = torch.randn(768, 768)
weight_q, scale = quantize_tensor_symmetric(weight, bits=4)
weight_approx = dequantize_tensor(weight_q, scale)
print(f"平均絶対誤差: {(weight - weight_approx).abs().mean():.6f}")
