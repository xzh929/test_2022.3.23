from torch import nn
import torch


class downsample(nn.Module):
    def __init__(self, c_in, c_out,stride=2):
        super(downsample, self).__init__()
        self.stride = stride
        self.down = nn.Sequential(
            nn.Conv2d(c_in, c_out, (1, 1), self.stride, bias=False),
            nn.BatchNorm2d(c_out)
        )

    def forward(self, x):
        return self.down(x)


class Bottleneck(nn.Module):
    def __init__(self, c_in, c_out, last_out, stride=2, is_first=False):
        super(Bottleneck, self).__init__()
        self.stride = stride
        self.is_first = is_first
        if is_first:
            self.basic = nn.Sequential(
                nn.Conv2d(last_out, c_in, (1, 1), (1, 1), bias=False),
                nn.BatchNorm2d(c_in),
                nn.Conv2d(c_in, c_in, (3, 3), self.stride, padding=(1, 1), bias=False),
                nn.BatchNorm2d(c_in),
                nn.Conv2d(c_in, c_out, (1, 1), (1, 1), bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU()
            )
            self.down = downsample(last_out, c_out,stride=self.stride)
        else:
            self.basic = nn.Sequential(
                nn.Conv2d(c_out, c_in, (1, 1), (1, 1), bias=False),
                nn.BatchNorm2d(c_in),
                nn.Conv2d(c_in, c_in, (3, 3), (1, 1), padding=(1, 1), bias=False),
                nn.BatchNorm2d(c_in),
                nn.Conv2d(c_in, c_out, (1, 1), (1, 1), bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU()
            )

    def forward(self, x):
        if self.is_first:
            return self.basic(x) + self.down(x)
        else:
            return self.basic(x) + x


class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(3, 64, (7, 7), (2, 2), padding=(3, 3), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1),
            Bottleneck(64, 256, 64, stride=1, is_first=True),
            Bottleneck(64, 256, 64),
            Bottleneck(64, 256, 64),
            Bottleneck(128, 512, 256, is_first=True),
            Bottleneck(128, 512, 256),
            Bottleneck(128, 512, 256),
            Bottleneck(128, 512, 256),
            Bottleneck(256, 1024, 512, is_first=True),
            Bottleneck(256, 1024, 512),
            Bottleneck(256, 1024, 512),
            Bottleneck(256, 1024, 512),
            Bottleneck(256, 1024, 512),
            Bottleneck(256, 1024, 512),
            Bottleneck(512, 2048, 1024, is_first=True),
            Bottleneck(512, 2048, 1024),
            Bottleneck(512, 2048, 1024),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.out_layer = nn.Sequential(
            nn.Linear(2048, 10)
        )

    def forward(self, x):
        cnn_out = self.layer(x)
        cnn_out = cnn_out.reshape(-1, 2048)
        out = self.out_layer(cnn_out)
        return out


if __name__ == '__main__':
    a = torch.randn(2, 3, 32, 32)
    net = Bottleneck(64, 256, 64, stride=1, is_first=True)
    net2 = Bottleneck(64, 256, 64)
    net1 = ResNet50()
    down = nn.Conv2d(64, 256, (1, 1), 1)
    # x = down(a)
    # y = net(a)
    print(net1)
    print(net1(a).shape)
    # out = x+y
    # print(net)
    # print(x.shape)
    # print(y.shape)
    # print(out.shape)
    # print(net2)
