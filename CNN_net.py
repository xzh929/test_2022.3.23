from torch import nn
import torch

class CNN_net(nn.Module):
    def __init__(self):
        super(CNN_net,self).__init__()
        self.cnn_layer = nn.Sequential(
            nn.Conv2d(1,16,3,1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, 1)
        )

        self.mlp_layer = nn.Sequential(
            nn.Linear(256*18*18,10),
            nn.Softmax(dim=1)
        )

    def forward(self,input_x):
        cnn_out = self.cnn_layer(input_x)
        cnn_out = cnn_out.reshape(-1,256*18*18)
        out = self.mlp_layer(cnn_out)
        return out

if __name__ == '__main__':
    a = torch.randn(1,1,28,28)
    net = CNN_net()
    out = net(a)
    print(out.shape)

