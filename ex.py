import torch

#1차 텐서
a = torch.ones(3)
print(f"1D 텐서: {a}, Shape: {a.shape}")

#2차원 텐서
points = torch.tensor([[4.0,1.0], [5.0,3.0],[2.0,1.0]])
print(f"2D 텐서:\n{points}, Shape: {points.shape}")

#3차원 텐서
img_t = torch.randn(3,5,5)
print(f"3D 텐서 (이미지): Shape: {img_t.shape}")


