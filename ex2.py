import torch

t_u = torch.tensor([35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4])
print(f"Original t_u: {t_u}")
print(f"Original t_u mean: {t_u.mean():.2f}, std: {t_u.std():.2f}")

# 단순하게 스케일링
t_un = 0.1 * t_u
print(f"Scaled t_un: {t_un}")
print(f"Scaled t_un mean: {t_un.mean():.2f}, std: {t_un.std():.2f}")

# Z-점수 정규화
mean = t_u.mean()
std = t_u.std()
t_uz = (t_u - mean) / std
print(f"Z-score normalized t_uz: {t_uz}")
print(f"Z-score t_uz mean: {t_uz.mean():.2f}, std: {t_uz.std():.2f}")

