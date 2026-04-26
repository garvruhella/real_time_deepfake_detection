import torch
import torch.nn as nn

# 128x128 Discriminator for DCGAN
class DCGANDiscriminator128(nn.Module):
    def __init__(self, in_channels=3, d_channels=64):
        super(DCGANDiscriminator128, self).__init__()
        self.net = nn.Sequential(
            # Input: 3 x 128 x 128
            nn.Conv2d(in_channels, d_channels, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 64 x 64 x 64

            nn.Conv2d(d_channels, d_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 128 x 32 x 32

            nn.Conv2d(d_channels * 2, d_channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 256 x 16 x 16

            nn.Conv2d(d_channels * 4, d_channels * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # 512 x 8 x 8

            nn.Conv2d(d_channels * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()  # outputs probability [0, 1]
        )

    def forward(self, x):
        return self.net(x).view(-1, 1).squeeze(1)
