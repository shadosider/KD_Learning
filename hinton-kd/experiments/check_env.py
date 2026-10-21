import torch
import torchvision

print("PyTorch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("Device:", "cuda" if torch.cuda.is_available() else "cpu")

x = torch.randn(4, 10)
print("Tensor shape:", x.shape)
print("Environment works.")
