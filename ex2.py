# import torch
#
# t_u = torch.tensor([35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4])
# print(f"Original t_u: {t_u}")
# print(f"Original t_u mean: {t_u.mean():.2f}, std: {t_u.std():.2f}")
#
# # 단순하게 스케일링
# t_un = 0.1 * t_u
# print(f"Scaled t_un: {t_un}")
# print(f"Scaled t_un mean: {t_un.mean():.2f}, std: {t_un.std():.2f}")
#
# # Z-점수 정규화
# mean = t_u.mean()
# std = t_u.std()
# t_uz = (t_u - mean) / std
# print(f"Z-score normalized t_uz: {t_uz}")
# print(f"Z-score t_uz mean: {t_uz.mean():.2f}, std: {t_uz.std():.2f}")
#


#
# import torch
#
# def model(t_u, w, b):
#     return w * t_u + b
#
# def loss_fn(t_p, t_c):
#     return ((t_p - t_c)**2).mean()
#
# t_u = torch.tensor([35.7, 55.9]) # 입력 데이터
# t_c = torch.tensor([0.5, 14.0]) # 실제 정답
#
# params = torch.tensor([1.0, 0.0], requires_grad=True)
#
# # 순전파
# t_p = model(t_u, *params)
# loss = loss_fn(t_p, t_c)
#
# # 역전파
# loss.backward()
#
# print(f"Parameters: {params}")
# print(f"Gradients: {params.grad}")



# import torch
# import torch.optim as optim
#
# def model(t_u, w, b):
#     return w * t_u + b
#
# def loss_fn(t_p, t_c):
#     return ((t_p - t_c)**2).mean()
#
# t_u_norm = torch.tensor([3.57, 5.59, 5.82, 8.19, 5.63, 4.89, 3.39, 2.18, 4.84, 6.04, 6.84]) # 정규화된 입력
# t_c = torch.tensor([0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0])
#
# # 매개변수 -> 옵티의 객체로
# params = torch.tensor([1.0, 0.0], requires_grad=True)
# learning_rate = 1e-2
#
# optimizer = optim.SGD([params], lr=learning_rate)
# # 학습
# for epoch in range(1, 101):
#     t_p = model(t_u_norm, *params)
#     loss = loss_fn(t_p, t_c)
#
#     optimizer.zero_grad() # 기울기 0으로 초기화
#     loss.backward()       # backpropagation
#     optimizer.step()      # 매개변수 업데이트
#
#     if epoch % 10 == 0:
#         print(f"Epoch {epoch}, Loss {loss.item():.4f}, Params: {params.detach().numpy()}")
#
# # 최종 매개변수
# print(f"\nFinal Parameters: {params.detach().numpy()}")

#
# import torch
# from torch.utils.data import DataLoader, TensorDataset
#
# # 예시 데이터
# t_u = torch.randn(100, 1) # 100개 샘플, 1개 특성
# t_c = 2 * t_u + 3 + torch.randn(100, 1) * 0.1
#
# dataset = TensorDataset(t_u, t_c)
#
# # 배치 크기 10
# batch_size = 10
#
# # 데이터로더 생성
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
# n_epochs = 10
#
# # 학습
# for epoch in range(n_epochs):
#     for i, (input_batch, target_batch) in enumerate(dataloader):
#         # input_batch와 target_batch로 순전파, 손실, 역전파, 매개변수 업데이트 수행
#         # ...
#         print(f"Epoch: {epoch+1}, Iteration: {i+1}, Batch size: {input_batch.shape[0]}")
#     print(f"--- Epoch {epoch+1} completed ---")


import torch

def model(t_u, w, b):
    return w * t_u + b

def loss_fn(t_p, t_c):
    return ((t_p - t_c)**2).mean()

params = torch.tensor([1.0, 0.0], requires_grad=True)
t_u_val = torch.tensor([10.0, 20.0])
t_c_val = torch.tensor([5.0, 15.0])

# 검증 손실 계산 -> 기울기 추적 필요 없음
with torch.no_grad():
    val_t_p = model(t_u_val, *params)
    val_loss = loss_fn(val_t_p, t_c_val)
    print(f"Validation Loss: {val_loss.item()}")
    print(f"Validation loss requires_grad: {val_loss.requires_grad}")

# 훈련 손실 계산 시 (기울기 추적 필요)
train_t_u = torch.tensor([1.0, 2.0])
train_t_c = torch.tensor([0.0, 1.0])
train_t_p = model(train_t_u, *params)
train_loss = loss_fn(train_t_p, t_c_val)
print(f"Training loss requires_grad: {train_loss.requires_grad}")
