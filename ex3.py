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

# import torch
# import torch.nn as nn
#
# # 입력 1개, 출력 1개 linear (in, out)
# linear_model = nn.Linear(1, 1)
#
# # 입력 데이터 (배치 차원 11, 특징 1)
# t_u_val = torch.tensor([35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]).unsqueeze(1)
#
# # 모델에 입력 데이터 전달
# output = linear_model(t_u_val)
# print("Output tensor shape:", output.shape)
#
# # 모델의 가중치와 편향 확인
# print("Weight:", linear_model.weight)
# print("Bias:", linear_model.bias)
#

# import torch
# t_u = torch.tensor([35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4])
#
# # unsqueeze(1)를 사용하여 1차원 텐서에 피쳐 차원 추가
# t_u_batched = t_u.unsqueeze(1)
#
# print("Original t_u shape:", t_u.shape)
# print("Batched t_u shape:", t_u_batched.shape)

#
# import torch
# import torch.nn as nn
# from collections import OrderedDict
#
# # 1. 모듈 목록으로 정의
# seq_model_list = nn.Sequential(
#     nn.Linear(1, 13), # 첫 번째 선형 레이어
#     nn.Tanh(),        # Tanh 활성화 함수
#     nn.Linear(13, 1)  # 두 번째 선형 레이어
# )
# print("Model defined with list:\n", seq_model_list)
#
# # 2. OrderedDict로 정의 (이름 지정 가능)
# seq_model_ordered_dict = nn.Sequential(OrderedDict([
#     ('hidden_linear', nn.Linear(1, 8)),        # 은닉 선형 레이어
#     ('hidden_activation', nn.Tanh()),                              # 활성화 함수
#     ('output_linear', nn.Linear(8, 1))         # 출력 선형 레이어
# ]))
# print("\nModel defined with OrderedDict:\n", seq_model_ordered_dict)
# print("\nParameters of OrderedDict model:")
# for name, param in seq_model_ordered_dict.named_parameters():
#     print(name, param.shape)


import torch
import torch.nn as nn
from collections import OrderedDict

seq_model = nn.Sequential(OrderedDict([
    ('hidden_linear', nn.Linear(1, 8)),
    ('hidden_activation', nn.Tanh()),
    ('output_linear', nn.Linear(8, 1))
]))

print("--- Using .parameters() ---")
# .parameters()는 제너레이터를 반환 -> 그래서 list
for param in seq_model.parameters():
    print("Shape:", param.shape)
    print("Parameter:", param)

print("\n--- Using .named_parameters() ---")
for name, param in seq_model.named_parameters():
    print(f"Name: {name}, Shape: {param.shape}")
    print("Parameter:", param)

# 특정 매개변수에 직접 접근
print("\nAccessing specific parameter:")
print(seq_model.hidden_linear.weight)
print(seq_model.output_linear.bias)





