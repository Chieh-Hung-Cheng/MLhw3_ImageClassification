import torch
import torch.nn as nn
from Config import Config


class ConvBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, 5),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.MaxPool2d((2,2))
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
            ConvBlock(3, 16),
            ConvBlock(16, 8),
            ConvBlock(8, 4),
            nn.Flatten(),
            FullyConnectedBlock(14400,256),
            FullyConnectedBlock(256,16),
            FullyConnectedBlock(16,11),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # N,C,H,W = N,3,512,512
        return self.net(x)

if __name__ == "__main__":
    x_b = torch.rand((32,3,512,512))
    x_b = x_b.to(Config.device)
    model = ImageConvNet().to(Config.device)
    print(model(x_b).shape)
    print(torch.sum(model(x_b)[10]))
