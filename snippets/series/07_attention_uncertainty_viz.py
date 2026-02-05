import matplotlib.pyplot as plt
import numpy as np

def visualize_attention_with_uncertainty(sentence, attention_weights, kappa_values):
    """確信度付きAttentionの可視化
    
    Args:
        sentence: 単語のリスト
        attention_weights: Attention重み [seq_len, seq_len]
        kappa_values: 各トークンのκ値 [seq_len]
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左：Attentionヒートマップ
    ax1 = axes[0]
    im = ax1.imshow(attention_weights, cmap='Blues')
    ax1.set_xticks(range(len(sentence)))
    ax1.set_yticks(range(len(sentence)))
    ax1.set_xticklabels(sentence, rotation=45, ha='right')
    ax1.set_yticklabels(sentence)
    ax1.set_xlabel('Key')
    ax1.set_ylabel('Query')
    ax1.set_title('Attention Weights')
    plt.colorbar(im, ax=ax1)
    
    # 右：確信度バー
    ax2 = axes[1]
    colors = plt.cm.RdYlGn(kappa_values / kappa_values.max())
    bars = ax2.barh(range(len(sentence)), kappa_values, color=colors)
    ax2.set_yticks(range(len(sentence)))
    ax2.set_yticklabels(sentence)
    ax2.set_xlabel('Confidence (κ)')
    ax2.set_title('Per-token Confidence')
    ax2.axvline(x=10, color='r', linestyle='--', label='OOD threshold')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('attention_with_uncertainty.png', dpi=150)
    plt.close()
    
    print("Saved: attention_with_uncertainty.png")


# サンプルデータで実行
sentence = ["The", "quantum", "cat", "sat", "sleepily"]
np.random.seed(42)

# Attentionは「cat」と「sat」に集中するパターン
attention = np.array([
    [0.3, 0.1, 0.3, 0.2, 0.1],
    [0.1, 0.4, 0.2, 0.2, 0.1],
    [0.2, 0.1, 0.4, 0.2, 0.1],
    [0.1, 0.1, 0.3, 0.4, 0.1],
    [0.2, 0.1, 0.2, 0.2, 0.3],
])

# 確信度：一般的な単語は高κ、専門用語「quantum」は低κ
kappa = np.array([45.0, 8.0, 50.0, 55.0, 35.0])

visualize_attention_with_uncertainty(sentence, attention, kappa)
