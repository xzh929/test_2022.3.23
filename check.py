from torchvision import models
from torch import nn
import torch
from torch import optim

net = models.resnet18()
net2 = models.resnet50()
net3 = models.mobilenet_v2()
# print(net)
a = torch.randn(1,3,33,33)
# layer = nn.Conv2d(3, 64, (7, 7), (2, 2), padding=3)
# x = layer(a)
# print(net2(a).shape)
print(net2)