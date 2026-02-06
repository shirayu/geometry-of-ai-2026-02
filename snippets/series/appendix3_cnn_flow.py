# CNNの典型的な計算フロー（教育用の最小スタブ）
input_image = "dummy_image"


def conv1(t):
    return t


def conv2(t):
    return t


def relu(t):
    return t


x = input_image  # どんな画像でも
x = conv1(x)  # 常に conv1 が適用される
x = relu(x)
x = conv2(x)  # 常に conv2 が適用される
x = relu(x)
# ... 入力に関わらず同じ経路
