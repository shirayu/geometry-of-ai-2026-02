import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA

# 手書き数字データ（64次元）
digits = load_digits()
X, y = digits.data, digits.target

# PCAで2次元に削減
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 累積寄与率
cumsum = np.cumsum(pca.explained_variance_ratio_)
print(f"2成分での累積寄与率: {cumsum[1]:.2%}")
# → 約28%程度。72%の分散が「失われている」

# 可視化
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="tab10", alpha=0.5, s=10)
plt.colorbar(scatter, label="Digit")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA: 64D → 2D（約28%の分散を保持）")

plt.subplot(1, 2, 2)
pca_full = PCA().fit(X)
plt.plot(np.cumsum(pca_full.explained_variance_ratio_))
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("累積寄与率")
plt.axhline(y=0.9, color="r", linestyle="--", label="90%")
plt.legend()

plt.tight_layout()
plt.show()
