import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_blobs

# 2クラスのデータ生成
X, y = make_blobs(n_samples=100, centers=2, random_state=42, cluster_std=1.5)

# 線形SVM
svm = SVC(kernel='linear', C=1.0)
svm.fit(X, y)

# 決定境界の可視化
plt.figure(figsize=(8, 6))

# データ点
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k', s=50)

# サポートベクターを強調
plt.scatter(svm.support_vectors_[:, 0], svm.support_vectors_[:, 1], 
            s=200, facecolors='none', edgecolors='green', linewidths=2,
            label=f'Support Vectors (n={len(svm.support_vectors_)})')

# 決定境界とマージン
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

xx = np.linspace(xlim[0], xlim[1], 100)
yy = np.linspace(ylim[0], ylim[1], 100)
XX, YY = np.meshgrid(xx, yy)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = svm.decision_function(xy).reshape(XX.shape)

# 決定境界（実線）とマージン（破線）
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], 
           linestyles=['--', '-', '--'], linewidths=[1, 2, 1])

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('SVM: マージン最大化の幾何学')
plt.legend()
plt.show()
