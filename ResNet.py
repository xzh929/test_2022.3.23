from torch import nn


class Res_Block(nn.Module):
    def __init__(self, c_in, is_first=True):
        super(Res_Block, self).__init__()
        self.stride = (1, 1) if is_first else (2, 2)
        self.res_layer = nn.Sequential(
            nn.Conv2d(c_in, c_in, (3, 3), self.stride, padding=(1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(c_in, c_in, (3, 3), self.stride, padding=(1, 1), bias=False),
            nn.BatchNorm2d(64)
        )

    def forward(self, x):
        return self.res_layer(x) + x


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(3, 64, (7, 7), (2, 2), padding=(3, 3), bias=False)
        )
