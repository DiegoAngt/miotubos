"""Modelo 1: encoder-decoder sin conexiones de salto."""

from torch import nn

from .blocks import bloque


class EncoderDecoder(nn.Module):
    """Encoder-decoder pequeño sin conexiones de salto."""

    def __init__(self):
        super().__init__()
        self.enc1 = bloque(1, 16)
        self.enc2 = bloque(16, 32)
        self.centro = bloque(32, 64)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = bloque(32, 32)
        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec1 = bloque(16, 16)
        self.salida = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        centro = self.centro(self.pool(e2))
        d2 = self.dec2(self.up2(centro))
        d1 = self.dec1(self.up1(d2))
        return self.salida(d1)
