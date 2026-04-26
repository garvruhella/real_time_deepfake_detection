# src/model_generator.py
import torch
import torch.nn as nn

class DCGANGenerator128(nn.Module):
    """
    Standard DCGAN generator that maps a random noise vector z
    into a 128x128 RGB image using transposed convolutions.
    """
    def __init__(self, z_dim=100, ngf=64):
        super().__init__()
        self.net = nn.Sequential(
            # Input: (N, z_dim, 1, 1)
            nn.ConvTranspose2d(z_dim, ngf*16, 4, 1, 0, bias=False),  # 4x4
            nn.BatchNorm2d(ngf*16), nn.ReLU(True),

            nn.ConvTranspose2d(ngf*16, ngf*8, 4, 2, 1, bias=False),  # 8x8
            nn.BatchNorm2d(ngf*8), nn.ReLU(True),

            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),   # 16x16
            nn.BatchNorm2d(ngf*4), nn.ReLU(True),

            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),   # 32x32
            nn.BatchNorm2d(ngf*2), nn.ReLU(True),

            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),     # 64x64
            nn.BatchNorm2d(ngf), nn.ReLU(True),

            nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False),         # 128x128
            nn.Tanh()  # outputs in range [-1, 1]
        )

    def forward(self, z):
        # z: shape (batch, z_dim)
        z = z.view(z.size(0), z.size(1), 1, 1)
        return self.net(z)
