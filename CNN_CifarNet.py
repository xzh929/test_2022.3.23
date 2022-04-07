from torch import nn
import torch


class Res_net(nn.Module):
    def __init__(self, channel):
        super(Res_net, self).__init__()
        self.res_net = nn.Sequential(
            nn.Conv2d(channel, channel // 2, 3, 1, padding=1, bias=False),
            nn.BatchNorm2d(channel // 2),
            nn.ReLU(),
            nn.Conv2d(channel // 2, channel // 2, 3, 1, padding=1, bias=False),
            nn.BatchNorm2d(channel // 2),
            nn.ReLU(),
            nn.Conv2d(channel // 2, channel, 3, 1, padding=1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU()
        )

    def forward(self, input_x):
        out = self.res_net(input_x)
        return out + input_x


class Pool(nn.Module):
    def __init__(self, chan_in, chan_out):
        super(Pool, self).__init__()
        self.pool = nn.Sequential(
            nn.Conv2d(chan_in, chan_out, 3, 2, padding=1, bias=False),
            nn.BatchNorm2d(chan_out),
            nn.ReLU()
        )

    def forward(self, input_x):
        out = self.pool(input_x)
        return out


# 构建cifar卷积网络
class CNN_net(nn.Module):
    def __init__(self):
        super(CNN_net, self).__init__()
        self.cnn_layer = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1),
            nn.ReLU(),
            Res_net(16),
            Res_net(16),
            Res_net(16),

            Pool(16, 32),
            Res_net(32),
            Res_net(32),
            Res_net(32),
            Res_net(32),

            Pool(32, 64),
            Res_net(64),
            Res_net(64),
            Res_net(64),
            Res_net(64),
            Res_net(64),
            Res_net(64),

            Pool(64, 128),
            Res_net(128),
            Res_net(128),
            Res_net(128),
            Res_net(128),
            Res_net(128),
            Res_net(128),
            nn.Conv2d(128, 10, 4, 1)
        )

        # self.out_layer = nn.Sequential(
        #     nn.Linear(128 * 3 * 3, 10)
        # )

    def forward(self, input_X):
        cnn_out = self.cnn_layer(input_X)
        cnn_out = cnn_out.squeeze()
        # cnn_out = cnn_out.reshape(-1, 128 * 3 * 3)
        # out = self.out_layer(cnn_out)
        return cnn_out


if __name__ == '__main__':
    a = torch.randn(2, 3, 32, 32)
    net = CNN_net()
    out = net(a)
    # print(net)
    print(out.shape)
