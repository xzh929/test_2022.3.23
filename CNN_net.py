from torch import nn

class CNN_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_layer = nn.Sequential(
            nn.Conv2d(3,16,3,1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 1),
            nn.ReLU(),
            nn.Conv2d(3, 16, 3, 1),
            nn.ReLU(),
        )
