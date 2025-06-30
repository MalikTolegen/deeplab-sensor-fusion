from torch import nn

from config import RESIZE_SCALE
# 6개


class BaseSensor(nn.Module):
    def __init__(self, channels=2048):
        super(BaseSensor, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(6, 4225),
            nn.BatchNorm1d(4225),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.layer1(x)
        B, _ = x.shape
        x = x.unsqueeze(1).reshape(B, 1, RESIZE_SCALE, RESIZE_SCALE)
        x = self.layer2(x)

        return x