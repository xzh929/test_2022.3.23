from torch import nn
import torch


# 构建cifar卷积网络
class CNN_net(nn.Module):
    def __init__(self):
        super(CNN_net, self).__init__()
        self.cnn_layer = nn.Sequential(
            nn.Conv2d(3, 3, 3, 1),
            nn.ReLU(),
            nn.Conv2d(3, 6, 3, 1),
            nn.ReLU(),
            nn.Conv2d(6, 12, 3, 1),
            nn.ReLU(),
            nn.Conv2d(12, 24, 3, 1),
            nn.ReLU(),
            nn.Conv2d(24, 48, 3, 1)
        )

        self.out_layer = nn.Sequential(
            nn.Linear(48 * 22 * 22, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, input_X):
        cnn_out = self.cnn_layer(input_X)
        cnn_out = cnn_out.reshape(-1, 48 * 22 * 22)
        out = self.out_layer(cnn_out)
        return out


if __name__ == '__main__':
    a = torch.randn(1, 3, 32, 32)
    net = CNN_net()
    out = net(a)
    print(out.shape)
