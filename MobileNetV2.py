from torch import nn
import torch


class First_Conv(nn.Module):
    def __init__(self):
        super(First_Conv, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(),
        )

    def forward(self, x):
        return self.layer(x)


class Second_Conv(nn.Module):
    def __init__(self):
        super(Second_Conv, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1, bias=False, groups=32),
            nn.BatchNorm2d(32),
            nn.ReLU6(),
            nn.Conv2d(32, 16, 1, 1, bias=False),
            nn.BatchNorm2d(16)
        )

    def forward(self, x):
        return self.layer(x)


class Out_Conv(nn.Module):
    def __init__(self):
        super(Out_Conv, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(320, 1280, 1, 1, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU6()
        )

    def forward(self, x):
        return self.layer(x)


class InvertedResidual(nn.Module):
    def __init__(self, c_in, c_out, t, n, i, s):
        super(InvertedResidual, self).__init__()
        self.n = n
        self.s = s
        self.i = i
        self.c_mid = c_in
        if self.n == i+1:
            self.s = s
            self.c_mid = c_out
        else:
            self.s = 1
            self.c_mid = c_in
        self.layer = nn.Sequential(
            nn.Conv2d(c_in, t * c_in, (1, 1), 1, bias=False),
            nn.BatchNorm2d(t * c_in),
            nn.ReLU6(),
            nn.Conv2d(t * c_in, t * c_in, (3, 3), s, bias=False, padding=1, groups=t * c_in),
            nn.BatchNorm2d(t * c_in),
            nn.ReLU6(),
            nn.Conv2d(t * c_in, self.c_mid, (1, 1), 1, bias=False),
            nn.BatchNorm2d(self.c_mid)
        )

    def forward(self, x):
        if self.n == self.i+1:
            out = self.layer(x)
        else:
            out = self.layer(x) + x
        return out


class MobileNetV2(nn.Module):
    def __init__(self):
        super(MobileNetV2, self).__init__()
        self.first_conv = First_Conv()
        self.second_conv = Second_Conv()
        self.layer1 = self.make_layer(16, 24, 6, 2, 2)
        self.layer2 = self.make_layer(24, 32, 6, 3, 2)
        self.layer3 = self.make_layer(32, 64, 6, 4, 2)
        self.layer4 = self.make_layer(64, 96, 6, 3, 1)
        self.layer5 = self.make_layer(96, 160, 6, 3, 2)
        self.layer6 = self.make_layer(160, 320, 6, 1, 1)
        self.out_conv = Out_Conv()
        self.avgpool = nn.AvgPool2d(7, 1)
        self.drop = nn.Dropout2d(0.2)
        self.linear = nn.Linear(1280, 1000)

    def make_layer(self, c_in, c_out, t, n, s):
        layer = []
        for i in range(n):
            layer.append(InvertedResidual(c_in, c_out, t, n, i, s))
        return nn.Sequential(*layer)

    def forward(self, x):
        out = self.first_conv(x)
        out = self.second_conv(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.out_conv(out)
        out = self.avgpool(out)
        out = self.drop(out)
        out = self.linear(out)
        return out


if __name__ == '__main__':
    a = torch.randn(1, 3, 224, 224)
    net = MobileNetV2()
    print(net(a).shape)
