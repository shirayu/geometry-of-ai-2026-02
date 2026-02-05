import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import vonmises

theta = np.linspace(-np.pi, np.pi, 1000)
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

for ax, kappa in zip(axes, [1, 10, 100], strict=False):
    pdf = vonmises.pdf(theta, kappa)
    ax.plot(theta, pdf)
    ax.set_title(f"κ = {kappa}")
    ax.set_xlabel("θ")
    ax.set_ylabel("Density")

plt.tight_layout()
plt.show()
