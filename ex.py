import torch
import numpy as np
from torch.utils.data import dataloader

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



import torch

target = torch.tensor([6, 6, 7, 4, 5]) # 샘플이 5개
#점수가 0~9점이니 범주가 10개

# 0으로 채워진 텐서 하나면 1이어야함
target_onehot = torch.zeros(target.shape[0], 10)

# scatter_ 메서드를 사용하여 원-핫 인코딩 수행
target_onehot.scatter_(1, target.unsqueeze(1), 1.0)

print(f"원본 점수: {target}")
print(f"원-핫 인코딩된 점수:\n{target_onehot}")