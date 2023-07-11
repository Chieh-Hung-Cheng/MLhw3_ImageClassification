import torch
import torch.nn as nn
from Config import Config


class ConvBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, 3, 1, 1),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0)
        )

    def forward(self, x):
        return self.block(x)


class FullyConnectedBlock(nn.Module):
    def __init__(self, input_features, output_features):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_features, output_features),
            nn.BatchNorm1d(output_features),
            nn.ReLU(),
            nn.Dropout()
        )

    def forward(self, x):
        return self.block(x)


class ImageConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(3, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
            ConvBlock(256, 512),
            ConvBlock(512, 512),
            nn.Flatten(),
            FullyConnectedBlock(8192,1024),
            FullyConnectedBlock(1024,512),
            FullyConnectedBlock(512,11)
        )

    def forward(self, x):
        # N,C,H,W = N,3,512,512
        return self.net(x)

if __name__ == "__main__":
    x_b = torch.rand((32,3,128,128))
    x_b = x_b.to(Config.device)
    model = ImageConvNet().to(Config.device)
    print(model(x_b).shape)
    print(torch.sum(model(x_b)[10]))
