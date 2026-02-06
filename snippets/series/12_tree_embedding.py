import torch
import torch.nn as nn
import torch.optim as optim


class PoincareEmbedding(nn.Module):
    """ポアンカレ円板への埋め込み"""

    def __init__(self, num_nodes, dim, init_scale=0.001):
        super().__init__()
        # 小さな値で初期化（中心付近からスタート）
        self.embeddings = nn.Parameter(torch.randn(num_nodes, dim) * init_scale)
        self.eps = 1e-5

    def forward(self, indices):
        emb = self.embeddings[indices]
        # ポアンカレ円板内に射影
        return self._project(emb)

    def _project(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True)
        max_norm = 1 - self.eps
        return torch.where(norm > max_norm, x / norm * max_norm, x)

    def distance(self, u, v):
        """ポアンカレ距離"""
        u_norm_sq = torch.clamp(torch.sum(u * u, dim=-1), max=1 - self.eps)
        v_norm_sq = torch.clamp(torch.sum(v * v, dim=-1), max=1 - self.eps)
        diff_norm_sq = torch.sum((u - v) ** 2, dim=-1)

        x = 2 * diff_norm_sq / ((1 - u_norm_sq) * (1 - v_norm_sq) + self.eps)
        return torch.log(1 + x + torch.sqrt(x * x + 2 * x + self.eps))


def train_tree_embedding(edges, num_nodes, dim=2, epochs=1000, lr=0.01):
    """木構造をポアンカレ円板に埋め込む

    Args:
        edges: エッジのリスト [(parent, child), ...]
        num_nodes: ノード数
        dim: 埋め込み次元
        epochs: エポック数
        lr: 学習率

    Returns:
        学習済み埋め込み
    """
    model = PoincareEmbedding(num_nodes, dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    edges_tensor = torch.tensor(edges)

    for epoch in range(epochs):
        optimizer.zero_grad()

        # 正例：エッジで接続されたノードペア
        pos_u = model(edges_tensor[:, 0])
        pos_v = model(edges_tensor[:, 1])
        pos_dist = model.distance(pos_u, pos_v)

        # 負例：ランダムなノードペア
        neg_indices = torch.randint(0, num_nodes, (len(edges) * 5, 2))
        neg_u = model(neg_indices[:, 0])
        neg_v = model(neg_indices[:, 1])
        neg_dist = model.distance(neg_u, neg_v)

        # 損失：正例の距離を小さく、負例の距離を大きく
        margin = 1.0
        loss = torch.mean(torch.relu(pos_dist - neg_dist + margin))

        loss.backward()

        # 勾配クリップ（双曲空間では重要）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # 射影（境界内に保持）
        with torch.no_grad():
            model.embeddings.data = model._project(model.embeddings.data)

        if (epoch + 1) % 200 == 0:
            print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

    return model


# 使用例：簡単な木構造
# 0 (root)
# ├── 1
# │   ├── 3
# │   └── 4
# └── 2
#     ├── 5
#     └── 6

edges = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)]
num_nodes = 7

model = train_tree_embedding(edges, num_nodes, dim=2, epochs=1000)

# 結果の確認
with torch.no_grad():
    embeddings = model(torch.arange(num_nodes))
    norms = torch.norm(embeddings, dim=1)
    print("\n埋め込み結果:")
    for i in range(num_nodes):
        print(f"  Node {i}: pos={embeddings[i].numpy()}, norm={norms[i]:.4f}")

# 期待される結果：
# - 根ノード(0)は中心に近い
# - 葉ノード(3,4,5,6)は境界に近い
# - 深さに応じてノルムが増加
