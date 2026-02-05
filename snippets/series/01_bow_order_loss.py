from collections import Counter

def bag_of_words(sentence):
    """文を単語の袋（頻度辞書）に変換"""
    words = sentence.lower().split()
    return Counter(words)

# 意味が異なるが、Bag-of-Wordsでは同一
sentence1 = "the cat chased the dog"
sentence2 = "the dog chased the cat"

bow1 = bag_of_words(sentence1)
bow2 = bag_of_words(sentence2)

print(f"文1: '{sentence1}'")
print(f"  → BoW: {dict(bow1)}")
print(f"文2: '{sentence2}'")
print(f"  → BoW: {dict(bow2)}")
print(f"BoWは同一か: {bow1 == bow2}")  # True

# 出力:
# 文1: 'the cat chased the dog'
#   → BoW: {'the': 2, 'cat': 1, 'chased': 1, 'dog': 1}
# 文2: 'the dog chased the cat'
#   → BoW: {'the': 2, 'dog': 1, 'chased': 1, 'cat': 1}
# BoWは同一か: True
