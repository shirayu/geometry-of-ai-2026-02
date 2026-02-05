import torch


def fgsm_attack(model, x, y, epsilon=0.01, clip_min=0.0, clip_max=1.0):
    """Fast Gradient Sign Method による敵対的サンプル生成

    Args:
        model: 攻撃対象のモデル
        x: 入力テンソル
        y: 正解ラベル
        epsilon: 摂動の大きさ
        clip_min, clip_max: データのレンジ（画像なら0-1、標準化済みなら別の値）
    """
    model.eval()  # 評価モードに設定
    x = x.clone().detach().requires_grad_(True)

    # 勾配の混入を防ぐ
    model.zero_grad(set_to_none=True)

    output = model(x)
    loss = torch.nn.functional.cross_entropy(output, y)
    loss.backward()

    # 勾配の符号方向に摂動
    x_adv = x + epsilon * x.grad.sign()

    # データのレンジにクリップ（データ依存なので引数で指定）
    x_adv = torch.clamp(x_adv, clip_min, clip_max)

    return x_adv.detach()
