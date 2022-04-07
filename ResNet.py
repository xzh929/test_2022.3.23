from torch import nn
import torch


# 下采样层
class downsample(nn.Module):
    def __init__(self, c_in, c_out):
        super(downsample, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(c_in, c_out, (1, 1), (2, 2), bias=False),
            nn.BatchNorm2d(c_out)
        )

    def forward(self, x):
        return self.down(x)


# 残差块
class BasicBlock(nn.Module):
    def __init__(self, c_in, c_out, is_first=True):
        super(BasicBlock, self).__init__()
        self.stride = (2, 2) if is_first else (1, 1)
        self.is_first = is_first
        if is_first:
            self.basic = nn.Sequential(
                nn.Conv2d(c_in, c_out, (3, 3), self.stride, padding=(1, 1), bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(),
                nn.Conv2d(c_out, c_out, (3, 3), (1, 1), padding=(1, 1), bias=False),
                nn.BatchNorm2d(c_out),
            )
            self.down = downsample(c_in, c_out)
        else:
            self.basic = nn.Sequential(
                nn.Conv2d(c_in, c_out, (3, 3), self.stride, padding=(1, 1), bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(),
                nn.Conv2d(c_out, c_out, (3, 3), (1, 1), padding=(1, 1), bias=False),
                nn.BatchNorm2d(c_out)
            )

    # 第一次输入和输出形状不一，用下采样做一次转换
    def forward(self, x):
        if self.is_first:
            return self.basic(x) + self.down(x)
        else:
            return self.basic(x) + x


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(3, 64, (7, 7), (2, 2), padding=(3, 3), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, padding=1),
            BasicBlock(64, 64, False),
            BasicBlock(64, 64, False),
            BasicBlock(64, 128, True),
            BasicBlock(128, 128, False),
            BasicBlock(128, 256, True),
            BasicBlock(256, 256, False),
            BasicBlock(256, 512, True),
            BasicBlock(512, 512, False),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.out_layer = nn.Sequential(
            nn.Linear(512, 1000)
        )

    def forward(self, x):
        cnn_out = self.layer(x)
        cnn_out = cnn_out.reshape(-1, 512)
        out = self.out_layer(cnn_out)
        return out


class ResNet34(nn.Module):
    def __init__(self):
        super(ResNet34, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(3, 64, (7, 7), (2, 2), padding=(3, 3), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, padding=1),
            BasicBlock(64, 64, False),
            BasicBlock(64, 64, False),
            BasicBlock(64, 64, False),
            BasicBlock(64, 128, True),
            BasicBlock(128, 128, False),
            BasicBlock(128, 128, False),
            BasicBlock(128, 128, False),
            BasicBlock(128, 256, True),
            BasicBlock(256, 256, False),
            BasicBlock(256, 256, False),
            BasicBlock(256, 256, False),
            BasicBlock(256, 256, False),
            BasicBlock(256, 256, False),
            BasicBlock(256, 512, True),
            BasicBlock(512, 512, False),
            BasicBlock(512, 512, False),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.out_layer = nn.Sequential(
            nn.Linear(512, 1000)
        )

    def forward(self, x):
        cnn_out = self.layer(x)
        cnn_out = cnn_out.reshape(-1, 512)
        out = self.out_layer(cnn_out)
        return out


if __name__ == '__main__':
    a = torch.randn(1, 3, 33, 33)
    # net = BasicBlock(64,64,True).forward(a)
    net1 = ResNet18()
    net2 = ResNet34()
    # print(net2)
    print(net1(a).shape)
    print(net2(a).shape)

