import numpy as np

for r in [1, 3, 5]:
    euclidean = 2 * np.pi * r
    hyperbolic = 2 * np.pi * np.sinh(r)
    print(f"r={r}: Euclidean={euclidean:.1f}, Hyperbolic={hyperbolic:.1f}")
