import torch
import torch.nn as nn

# 128x128 Generator for DCGAN
class DCGANGenerator128(nn.Module):
    def __init__(self, z_dim=100, g_channels=64, out_channels=3):
        super(DCGANGenerator128, self).__init__()
        self.net = nn.Sequential(
            # Input: Z latent vector
            nn.ConvTranspose2d(z_dim, g_channels * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(g_channels * 16),
            nn.ReLU(True),
            # (g_channels*16) x 4 x 4

            nn.ConvTranspose2d(g_channels * 16, g_channels * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(g_channels * 8),
            nn.ReLU(True),
            # (g_channels*8) x 8 x 8

            nn.ConvTranspose2d(g_channels * 8, g_channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(g_channels * 4),
            nn.ReLU(True),
            # (g_channels*4) x 16 x 16

            nn.ConvTranspose2d(g_channels * 4, g_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(g_channels * 2),
            nn.ReLU(True),
            # (g_channels*2) x 32 x 32

            nn.ConvTranspose2d(g_channels * 2, g_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(g_channels),
            nn.ReLU(True),
            # (g_channels) x 64 x 64

            nn.ConvTranspose2d(g_channels, out_channels, 4, 2, 1, bias=False),
            nn.Tanh()  # Output in range [-1, 1]
            # 3 x 128 x 128
        )

    def forward(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)  # reshape noise vector
        return self.net(z)
