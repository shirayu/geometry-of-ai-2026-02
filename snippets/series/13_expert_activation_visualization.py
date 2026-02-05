import matplotlib.pyplot as plt
import torch

def visualize_expert_activation(router_logits, top_k=2):
    """ルーター出力をヒートマップとして可視化"""
    # router_logits: [batch_size, num_experts]
    probs = torch.softmax(router_logits, dim=-1).detach().cpu().numpy()
    
    plt.figure(figsize=(12, 4))
    plt.imshow(probs, aspect='auto', cmap='viridis')
    plt.xlabel('Expert Index')
    plt.ylabel('Token Index')
    plt.colorbar(label='Routing Probability')
    plt.title(f'Expert Activation Pattern (Top-{top_k} routing)')
    plt.show()
