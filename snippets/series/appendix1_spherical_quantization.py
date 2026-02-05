# 概念的なコード
x_normalized = F.normalize(x, dim=-1)  # ノルム1に正規化
x_quantized = quantize(x_normalized)   # 量子化
# この時点で ||x_quantized|| ≠ 1 の可能性
x_renormalized = F.normalize(x_quantized, dim=-1)  # 再正規化
