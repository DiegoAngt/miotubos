"""Modelo 3: U-Net con bloques residuales."""

import torch
from torch import nn

from .blocks import BloqueResidual


class UNetResidual(nn.Module):
    """Misma profundidad y canales que la U-Net simple, con bloques residuales."""

    def __init__(self):
        super().__init__()
        self.enc1 = BloqueResidual(1, 16)
        self.enc2 = BloqueResidual(16, 32)
        self.centro = BloqueResidual(32, 64)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = BloqueResidual(64, 32)
        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec1 = BloqueResidual(32, 16)
        self.salida = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        centro = self.centro(self.pool(e2))
        d2 = self.dec2(torch.cat([self.up2(centro), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.salida(d1)
