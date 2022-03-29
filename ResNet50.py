from torch import nn
import torch


class downsample(nn.Module):
    def __init__(self, c_in, c_out):
        super(downsample, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(c_in, c_out, (1, 1), (2, 2), bias=False),
            nn.BatchNorm2d(c_out)
        )

    def forward(self, x):
        return self.down(x)


class Bottleneck(nn.Module):
    def __init__(self, c_in, c_out, last_out, is_first=True):
        super(Bottleneck, self).__init__()
        self.stride = (2, 2) if is_first else (1, 1)
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
            self.down = downsample(last_out, c_out)
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
            Bottleneck(64, 256, 64, True),
            Bottleneck(64, 256, 64, False),
            Bottleneck(64, 256, 64, False),
            Bottleneck(128, 512, 256, True),
            Bottleneck(128, 512, 256, False),
            Bottleneck(128, 512, 256, False),
            Bottleneck(128, 512, 256, False),
            Bottleneck(256, 1024, 512, True),
            Bottleneck(256, 1024, 512, False),
            Bottleneck(256, 1024, 512, False),
            Bottleneck(256, 1024, 512, False),
            Bottleneck(256, 1024, 512, False),
            Bottleneck(256, 1024, 512, False),
            Bottleneck(512, 2048, 1024, True),
            Bottleneck(512, 2048, 1024, False),
            Bottleneck(512, 2048, 1024, False),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.out_layer = nn.Sequential(
            nn.Linear(2048, 1000)
        )

    def forward(self, x):
        cnn_out = self.layer(x)
        cnn_out = cnn_out.reshape(-1, 2048)
        out = self.out_layer(cnn_out)
        return out


if __name__ == '__main__':
    a = torch.randn(1, 3, 100, 100)
    # net = Bottleneck(512, 2048, 1024, False)
    net1 = ResNet50()
    print(net1)
    # print(net1(a).shape)
