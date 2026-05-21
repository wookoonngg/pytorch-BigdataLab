# import torch
# import numpy as np
# from torch.utils.data import dataloader

# #1차 텐서
# a = torch.ones(3)
# print(f"1D 텐서: {a}, Shape: {a.shape}")
#
# #2차원 텐서
# points = torch.tensor([[4.0,1.0], [5.0,3.0],[2.0,1.0]])
# print(f"2D 텐서:\n{points}, Shape: {points.shape}")
#
# #3차원 텐서
# img_t = torch.randn(3,5,5)
# print(f"3D 텐서 (이미지): Shape: {img_t.shape}")
#
#

# points = torch.tensor([[4.0,1.0], [5.0,3.0], [2.0,1.0]])
#
# print(points.storage())
# print(points.storage_offset())
# print(points.size())
# print(points.stride())


# a = torch.ones(3, 2)
# print("Before zero_():")
# print(a)
#
# a.zero_() # 인플레이스 연산
# print("\nAfter zero_():")
# print(a)
#
# # 일반 연산
# b = torch.ones(2, 2)
# c = b.add(1.0) # 새로운 텐서들로 반환
# print("\nb (original):")
# print(b)
# print("\nc (new tensor):")
# print(c)
# print("b 안변함")
# print(b)



# 기본 dtype
# a = torch.ones(3)
# print(f"기본 dtype 텐서: {a.dtype}") # torch.float32
#
# # 명시적으로 float64 (double)
# double_points = torch.ones(10, 2, dtype=torch.double)
# print(f"double dtype 텐서: {double_points.dtype}") # torch.float64
#
# # int16 (short)
# short_points = torch.tensor([[1, 2], [3, 4]], dtype=torch.short)
# print(f"short dtype 텐서: {short_points.dtype}") # torch.int16
#
# # to() 변경
# converted_points = double_points.to(torch.float32)
# print(f"변환된 텐서 dtype: {converted_points.dtype}") # torch.float32

# for batch in dataloader:
#     data, label = batch
#     data = data.float()
#     batch /= 255.0
#
# n_channels = batch.shape[1]
# for c in range(n_channels):
#     mean = torch.mean(batch[:, c])
#     std = torch.std(batch[:, c])
#     batch[:, c] = (batch[:, c] - mean) / std
#



# import torch
#
# target = torch.tensor([6, 6, 7, 4, 5]) # 샘플이 5개
# #점수가 0~9점이니 범주가 10개
#
# # 0으로 채워진 텐서 하나면 1이어야함
# target_onehot = torch.zeros(target.shape[0], 10)
#
# # scatter_ 메서드를 사용하여 원-핫 인코딩 수행
# target_onehot.scatter_(1, target.unsqueeze(1), 1.0)
#
# print(f"원본 점수: {target}")
# print(f"원-핫 인코딩된 점수:\n{target_onehot}")


# import torch
# import torch.nn as nn
# import torch.optim as optim
#
# # -------------------------
# # 1. 간단한 데이터
# # -------------------------
# sentences = [
#     "i love machine learning",
#     "i love deep learning",
#     "i hate bugs",
#     "bugs are annoying"
# ]
#
# labels = [1, 1, 0, 0]  # 1: positive, 0: negative
#
# # -------------------------
# # 2. 단어 사전 만들기
# # -------------------------
# word2idx = {}
# idx = 0
#
# for sentence in sentences:
#     for word in sentence.split():
#         if word not in word2idx:
#             word2idx[word] = idx
#             idx += 1
#
# vocab_size = len(word2idx)
#
# # -------------------------
# # 3. 문장을 숫자로 변환
# # -------------------------
# def encode(sentence):
#     return [word2idx[word] for word in sentence.split()]
#
# encoded_sentences = [encode(s) for s in sentences]
#
# # 패딩 맞추기
# max_len = max(len(s) for s in encoded_sentences)
#
# def pad(seq):
#     return seq + [0] * (max_len - len(seq))
#
# padded = [pad(s) for s in encoded_sentences]
#
# X = torch.tensor(padded)
# y = torch.tensor(labels)
#
# # -------------------------
# # 4. 모델 정의
# # -------------------------
# class TextEmbeddingModel(nn.Module):
#     def __init__(self, vocab_size, embed_dim):
#         super().__init__()
#         self.embedding = nn.Embedding(vocab_size, embed_dim)
#         self.fc = nn.Linear(embed_dim, 2)
#
#     def forward(self, x):
#         emb = self.embedding(x)       # (batch, seq, dim)
#         emb = emb.mean(dim=1)         # 평균 pooling → 문장 벡터
#         out = self.fc(emb)
#         return out
#
# model = TextEmbeddingModel(vocab_size, embed_dim=10)
#
# # -------------------------
# # 5. 학습 설정
# # -------------------------
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.01)
#
# # -------------------------
# # 6. 학습
# # -------------------------
# for epoch in range(100):
#     optimizer.zero_grad()
#
#     outputs = model(X)
#     loss = criterion(outputs, y)
#
#     loss.backward()
#     optimizer.step()
#
#     if epoch % 10 == 0:
#         print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
#
# # -------------------------
# # 7. 임베딩 확인
# # -------------------------
# print("\n=== Word Embeddings ===")
# for word, idx in word2idx.items():
#     print(word, model.embedding.weight[idx].detach())





import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# -------------------------
# 1. 명확한 데이터 (핵심🔥)
# -------------------------
sentences = [
    "good happy nice",
    "good awesome happy",
    "nice good happy",
    "bad sad terrible",
    "terrible bad sad",
    "sad horrible bad"
]

labels = [1, 1, 1, 0, 0, 0]  # 1: positive, 0: negative

# -------------------------
# 2. 단어 사전
# -------------------------
word2idx = {}
idx = 0

for sentence in sentences:
    for word in sentence.split():
        if word not in word2idx:
            word2idx[word] = idx
            idx += 1

vocab_size = len(word2idx)

# -------------------------
# 3. 인코딩 + 패딩
# -------------------------
def encode(sentence):
    return [word2idx[word] for word in sentence.split()]

encoded = [encode(s) for s in sentences]
max_len = max(len(s) for s in encoded)

def pad(seq):
    return seq + [0] * (max_len - len(seq))

X = torch.tensor([pad(s) for s in encoded])
y = torch.tensor(labels)

# -------------------------
# 4. 모델
# -------------------------
class Model(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, 2)

    def forward(self, x):
        emb = self.embedding(x)
        emb = emb.mean(dim=1)  # 문장 벡터
        return self.fc(emb)

model = Model(vocab_size, embed_dim=5)

# -------------------------
# 5. 학습
# -------------------------
optimizer = optim.Adam(model.parameters(), lr=0.05)
criterion = nn.CrossEntropyLoss()

for epoch in range(200):
    optimizer.zero_grad()
    out = model(X)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# -------------------------
# 6. 임베딩 확인
# -------------------------
print("\n=== Embeddings ===")
for word, i in word2idx.items():
    print(word, model.embedding.weight[i].detach())

# -------------------------
# 7. 유사도 비교
# -------------------------
def sim(w1, w2):
    v1 = model.embedding.weight[word2idx[w1]]
    v2 = model.embedding.weight[word2idx[w2]]
    return F.cosine_similarity(v1, v2, dim=0)

print("\n=== Similarity ===")
print("good vs happy:", sim("good", "happy").item())
print("good vs bad:", sim("good", "bad").item())
print("bad vs terrible:", sim("bad", "terrible").item())




















