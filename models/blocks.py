"""Bloques de convolución que comparten los modelos."""

from torch import nn

GRUPOS = 8  # grupos de GroupNorm; divide a 16, 32 y 64 canales


def bloque(entrada, salida):
    """Bloque simple: dos convoluciones 3×3 con ReLU."""
    return nn.Sequential(
        nn.Conv2d(entrada, salida, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(salida, salida, 3, padding=1),
        nn.ReLU(),
    )


class BloqueResidual(nn.Module):
    """Dos convoluciones 3×3 más un atajo que suma la entrada a la salida."""

    def __init__(self, entrada, salida):
        super().__init__()
        # GroupNorm en lugar de BatchNorm porque el lote es de solo 2 recortes.
        self.conv1 = nn.Conv2d(entrada, salida, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(GRUPOS, salida)
        self.conv2 = nn.Conv2d(salida, salida, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(GRUPOS, salida)
        self.relu = nn.ReLU()
        # El atajo necesita una convolución 1×1 cuando cambian los canales.
        self.atajo = (
            nn.Identity() if entrada == salida
            else nn.Conv2d(entrada, salida, 1, bias=False)
        )

    def forward(self, x):
        y = self.relu(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        return self.relu(y + self.atajo(x))
