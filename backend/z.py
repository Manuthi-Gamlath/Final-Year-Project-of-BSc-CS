import torch

try:
    x = torch.randn(1024, 1024).cuda()
    y = torch.randn(1024, 1024).cuda()
    z = torch.mm(x, y)
    print("CUDA Test Passed")
except Exception as e:
    print("CUDA Test Failed:", e)

