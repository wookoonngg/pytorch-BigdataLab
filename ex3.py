# import torch
# import torch.nn as nn
#
# # 선형 모델 생성
# linear_model = nn.Linear(1, 1)
#
# # 모델의 매개변수 끌고오기 가능
# print("Weight:", linear_model.weight)
# print("Bias:", linear_model.bias)
#
# # 입력 텐서
# x = torch.ones(10, 1)
# output = linear_model(x)
# print("Output shape:", output.shape)

import torch
import torch.nn as nn

# 입력 1개, 출력 1개 linear (in, out)
linear_model = nn.Linear(1, 1)

# 입력 데이터 (배치 차원 11, 특징 1)
t_u_val = torch.tensor([35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]).unsqueeze(1)

# 모델에 입력 데이터 전달
output = linear_model(t_u_val)
print("Output tensor shape:", output.shape)

# 모델의 가중치와 편향 확인
print("Weight:", linear_model.weight)
print("Bias:", linear_model.bias)




