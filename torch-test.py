import torch

x = torch.rand(5, 3)
print(x)

print("Torch is available: ", torch.cuda.is_available())
