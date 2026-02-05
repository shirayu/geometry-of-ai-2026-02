import numpy as np
import matplotlib.pyplot as plt

distances = []
for _ in range(1000):
    x = np.random.randn(100)
    y = np.random.randn(100)
    distances.append(np.linalg.norm(x - y))

plt.hist(distances, bins=30)
plt.xlabel("Euclidean Distance")
plt.title("Distance distribution in 100D")
plt.show()
print(f"Mean: {np.mean(distances):.2f}, Std: {np.std(distances):.2f}")
