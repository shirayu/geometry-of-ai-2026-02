from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

def cosine_knn(X_train, X_query, k=5):
    """コサイン類似度ベースのk近傍探索
    
    L2正規化により、ユークリッド距離とコサイン類似度は単調同値になる：
    ||u - v||^2 = 2(1 - u・v)  （|u| = |v| = 1 のとき）
    
    注意: データが意味構造を持つ埋め込み空間である場合に有効
    """
    # L2正規化により、ユークリッド距離での近傍 = コサイン類似度での近傍
    X_train_norm = normalize(X_train)
    X_query_norm = normalize(X_query)
    
    nn = NearestNeighbors(n_neighbors=k, metric='cosine')
    nn.fit(X_train_norm)
    distances, indices = nn.kneighbors(X_query_norm)
    return distances, indices
