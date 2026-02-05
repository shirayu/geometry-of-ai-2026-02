import matplotlib.pyplot as plt
import numpy as np

angles = []
for _ in range(1000):
    x = np.random.randn(100)
    y = np.random.randn(100)
    cos_sim = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    cos_sim = np.clip(cos_sim, -1, 1)  # 数値誤差対策
    angles.append(np.degrees(np.arccos(cos_sim)))

plt.hist(angles, bins=30)
plt.xlabel("Angle (degrees)")
plt.title("Angle distribution in 100D")
plt.show()
print(f"Mean: {np.mean(angles):.1f}°, Std: {np.std(angles):.1f}°")
