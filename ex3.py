import torch
import torch.nn as nn

# 선형 모델 생성
linear_model = nn.Linear(1, 1)

# 모델의 매개변수 끌고오기 가능
print("Weight:", linear_model.weight)
print("Bias:", linear_model.bias)

# 입력 텐서
x = torch.ones(10, 1)
output = linear_model(x)
print("Output shape:", output.shape)