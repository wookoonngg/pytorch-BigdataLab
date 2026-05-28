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



import torch
import torch.optim as optim

def model(t_u, w, b):
    return w * t_u + b

def loss_fn(t_p, t_c):
    return ((t_p - t_c)**2).mean()

t_u_norm = torch.tensor([3.57, 5.59, 5.82, 8.19, 5.63, 4.89, 3.39, 2.18, 4.84, 6.04, 6.84]) # 정규화된 입력
t_c = torch.tensor([0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0])

# 매개변수 -> 옵티의 객체로
params = torch.tensor([1.0, 0.0], requires_grad=True)
learning_rate = 1e-2

optimizer = optim.SGD([params], lr=learning_rate)
# 학습
for epoch in range(1, 101):
    t_p = model(t_u_norm, *params)
    loss = loss_fn(t_p, t_c)

    optimizer.zero_grad() # 기울기 0으로 초기화
    loss.backward()       # backpropagation
    optimizer.step()      # 매개변수 업데이트

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss {loss.item():.4f}, Params: {params.detach().numpy()}")

# 최종 매개변수
print(f"\nFinal Parameters: {params.detach().numpy()}")