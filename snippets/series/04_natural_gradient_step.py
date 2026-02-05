import torch


def natural_gradient_step(params, loss_fn, data, lr=0.01, damping=1e-4):
    """自然勾配ステップの簡易実装（教育目的）

    警告: これは小規模モデルでの概念実証用。
    大規模モデルではK-FAC等の近似が必要。

    Args:
        params: モデルパラメータ
        loss_fn: 損失関数
        data: 入力データ
        lr: 学習率
        damping: 数値安定性のための正則化項
    """
    # 通常の勾配を計算
    loss = loss_fn(data)
    grads = torch.autograd.grad(loss, params, create_graph=True)

    # フィッシャー情報行列の対角近似（経験的フィッシャー）
    # 注: これは非常に粗い近似
    fisher_diag = []
    for g in grads:
        fisher_diag.append(g.detach() ** 2 + damping)

    # 自然勾配 = F^{-1} * grad
    natural_grads = []
    for g, f in zip(grads, fisher_diag):
        natural_grads.append(g.detach() / f)

    # パラメータ更新
    with torch.no_grad():
        for p, ng in zip(params, natural_grads):
            p.data -= lr * ng

    return loss.item()


# 注: 実用的な自然勾配の実装には、K-FAC (Martens & Grosse, 2015)
# や EKFAC (George et al., 2018) などの手法を参照されたい。
