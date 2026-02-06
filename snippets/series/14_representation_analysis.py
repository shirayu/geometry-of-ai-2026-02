import numpy as np
from ripser import ripser


def analyze_representation_topology(embeddings, labels, maxdim=1):
    """表現空間のトポロジーをクラスごとに解析

    Args:
        embeddings: 表現ベクトル (n_samples, n_features)
        labels: クラスラベル (n_samples,)
        maxdim: 計算する最大次元

    Returns:
        各クラスのトポロジー的特徴の要約
    """
    results = {}
    unique_labels = np.unique(labels)

    for label in unique_labels:
        mask = labels == label
        class_embeddings = embeddings[mask]

        # パーシステントホモロジー計算
        dgms = ripser(class_embeddings, maxdim=maxdim)["dgms"]

        # 各次元の持続性統計
        stats = {}
        for dim in range(maxdim + 1):
            if len(dgms[dim]) > 0:
                # 無限大の死亡時刻を除外
                finite_mask = np.isfinite(dgms[dim][:, 1])
                if finite_mask.any():
                    persistence = dgms[dim][finite_mask, 1] - dgms[dim][finite_mask, 0]
                    stats[f"H{dim}_count"] = len(persistence)
                    stats[f"H{dim}_max_persistence"] = persistence.max()
                    stats[f"H{dim}_mean_persistence"] = persistence.mean()
                else:
                    stats[f"H{dim}_count"] = 0

        results[label] = stats

    return results


# 使用例（モデルの表現を取得した後）
# embeddings = model.get_embeddings(data)
# topology_stats = analyze_representation_topology(embeddings, labels)
# for label, stats in topology_stats.items():
#     print(f"Class {label}: {stats}")
