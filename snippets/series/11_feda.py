import torch


def feda_transform(features, domain_label):
    """FEDA変換を適用

    Args:
        features: 元の特徴量 [batch_size, dim]
        domain_label: ドメインラベル ('source' または 'target')

    Returns:
        拡張された特徴量 [batch_size, 3*dim]
    """
    batch_size, dim = features.shape

    # 3つのブロックを準備
    common = features  # 共通ブロック
    source_specific = torch.zeros_like(features)  # ソース専用
    target_specific = torch.zeros_like(features)  # ターゲット専用

    if domain_label == "source":
        source_specific = features
    elif domain_label == "target":
        target_specific = features

    # [共通; ソース専用; ターゲット専用] として結合
    return torch.cat([common, source_specific, target_specific], dim=-1)


def batch_feda_transform(features, domain_labels):
    """バッチ単位でFEDA変換を適用

    Args:
        features: 元の特徴量 [batch_size, dim]
        domain_labels: ドメインラベルのテンソル [batch_size]
                      (0=source, 1=target)

    Returns:
        拡張された特徴量 [batch_size, 3*dim]
    """
    batch_size, dim = features.shape
    expanded = torch.zeros(batch_size, 3 * dim, device=features.device)

    # 共通ブロック（全サンプル共通）
    expanded[:, :dim] = features

    # ソース専用ブロック
    source_mask = domain_labels == 0
    expanded[source_mask, dim : 2 * dim] = features[source_mask]

    # ターゲット専用ブロック
    target_mask = domain_labels == 1
    expanded[target_mask, 2 * dim :] = features[target_mask]

    return expanded


# 使用例
batch_size, dim = 32, 128

# ソースドメインのデータ
source_features = torch.randn(batch_size // 2, dim)
source_labels = torch.zeros(batch_size // 2, dtype=torch.long)

# ターゲットドメインのデータ
target_features = torch.randn(batch_size // 2, dim)
target_labels = torch.ones(batch_size // 2, dtype=torch.long)

# バッチとして結合
all_features = torch.cat([source_features, target_features], dim=0)
all_domain_labels = torch.cat([source_labels, target_labels], dim=0)

# FEDA変換
expanded_features = batch_feda_transform(all_features, all_domain_labels)

print(f"Original feature shape: {all_features.shape}")
print(f"Expanded feature shape: {expanded_features.shape}")

# ソース専用ブロックとターゲット専用ブロックの構造的な分離を確認
# 各サンプルで、ソースブロックとターゲットブロックの一方は常に0になる
source_block = expanded_features[:, dim : 2 * dim]
target_block = expanded_features[:, 2 * dim :]
# ソースサンプルではtarget_blockが0、ターゲットサンプルではsource_blockが0
# なので、要素ごとの積の和は理論的に0になる
element_wise_product = (source_block * target_block).sum()
print(f"Element-wise product sum (structural separation): {element_wise_product.item():.6f}")

# この拡張された特徴量は、通常の分類器（SVMなど）に渡せる
# 分類器は自動的に、共通の特徴と各ドメイン固有の特徴を使い分ける
