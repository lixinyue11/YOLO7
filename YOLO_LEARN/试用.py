# import torch
# from torch import nn
#
#
# def _make_grid(nx=20, ny=20):
#     yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
#     print(torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)))
#     return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()
# a=_make_grid(nx=4, ny=4)
#
#
# from torchvision.models import resnet50
# import thop
# model = resnet50()
# input = torch.randn(1, 3, 224, 224)
# macs, params = thop.profile(model, inputs=(input,))
# print("FLOPs:", macs)  # FLOPs模型复杂度
# print("params:", params)  # 参数量
# thop.clever_format([macs, params], "%.3f")  # 格式化输出结果

import torch

t1=torch.Tensor([[200,200],[200,200],[200,200],[200,200]])
ar = t1[:, 1] / t1[:, 0]
a=t1.view(-1, 1, 1)
irect = ar.argsort()
print(t1[:, 1])
print(t1[:, 0])
print(irect)


grid = [torch.zeros(1)] * 3
a = torch.tensor([ [12,16, 19,36, 40,28],[36,75, 76,55, 72,146] ,[142,110, 192,243, 459,401]]).float().view(3, -1, 2)
b= a.clone().view(3, 1, -1, 1, 1, 2)
print(a)
print(b)

def _make_grid(nx=20, ny=20):
    yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
    return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()