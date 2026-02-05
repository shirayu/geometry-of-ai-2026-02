import numpy as np

def softmax(z, tau):
    z_scaled = z / tau
    exp_z = np.exp(z_scaled - np.max(z_scaled))  # 数値安定化
    return exp_z / np.sum(exp_z)

logits = np.array([2.0, 1.0, 0.1])

for tau in [0.1, 1.0, 10.0]:
    probs = softmax(logits, tau)
    print(f"τ={tau:4.1f}: {probs.round(4)}")
