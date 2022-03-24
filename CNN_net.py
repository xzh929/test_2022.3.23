from torch import nn
import torch


# 构建mnist卷积神经网络
class CNN_net(nn.Module):
    def __init__(self):
        super(CNN_net, self).__init__()
        self.cnn_layer = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1),  # 定义卷积层，参数分别为图片的通道，卷积核的数量，卷积核形状（3*3），步长为1
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, 1)
        )

        self.mlp_layer = nn.Sequential(  # 通过全连接接收卷积输出，转换为2维数据（批次，标签数量）
            nn.Linear(256 * 18 * 18, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, input_x):
        cnn_out = self.cnn_layer(input_x) # 卷积网络接收数据
        cnn_out = cnn_out.reshape(-1, 256 * 18 * 18)    # 将卷积输出数据形状转换为全连接可接收的2维数据
        out = self.mlp_layer(cnn_out)   # 将转换的数据传入全连接网络
        return out


if __name__ == '__main__':
    a = torch.randn(1, 1, 28, 28)
    net = CNN_net()
    out = net(a)
    print(out.shape)
