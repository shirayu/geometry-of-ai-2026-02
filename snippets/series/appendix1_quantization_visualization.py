import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE


def visualize_quantization_effect(hidden_orig, hidden_quant):
    """量子化前後の中間表現（テンソル）を比較する簡易例"""
    combined = torch.cat([hidden_orig, hidden_quant], dim=0)
    tsne = TSNE(n_components=2, perplexity=30)
    embedded = tsne.fit_transform(combined.cpu().numpy())

    n = len(hidden_orig)
    plt.scatter(embedded[:n, 0], embedded[:n, 1], label="Original", alpha=0.5)
    plt.scatter(embedded[n:, 0], embedded[n:, 1], label="Quantized", alpha=0.5)
    plt.legend()
    plt.title("Representation Space: Original vs Quantized")
    plt.show()
