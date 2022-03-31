from torch import nn
from torch.nn.functional import interpolate
import torch


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(7, 1)
        )
        self.out_layer = nn.Sequential(
            nn.Linear(32, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        out = self.layer(x)
        out = out.reshape(-1, 32)
        return self.out_layer(out)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.layer1 = nn.Conv2d(32, 16, 3, 1,padding=1)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Conv2d(16, 1, 3, 1,padding=1)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        out = self.layer1(x)
        out = self.relu1(out)
        out = interpolate(out, scale_factor=2, mode='nearest')
        out = self.layer2(out)
        out = self.relu1(out)
        out = interpolate(out, scale_factor=2, mode='nearest')
        return out


if __name__ == '__main__':
    a = torch.randn(1, 1, 28, 28)
    # layer1 = nn.Conv2d(1, 16, 3, 2, padding=1, bias=False)
    # layer2 = nn.Conv2d(16, 32, 3, 2, padding=1, bias=False)
    # out = layer1(a)
    # out = layer2(out)
    # print(out.shape)
    net1 = Encoder()
    net2 = Decoder()
    out1 = net1(a)
    layer = nn.Linear(10, 32)
    out1 = layer(out1)
    out1 = out1.reshape(1, 32, 1, 1)
    out1 = interpolate(out1,scale_factor=7,mode='nearest')
    out2 = net2(out1)
    print(out2.shape)
