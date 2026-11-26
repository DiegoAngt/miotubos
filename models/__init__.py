"""Modelos disponibles. El nombre es el que se pasa a train.py
y también el de su carpeta dentro de results/."""

from .encoder_decoder import EncoderDecoder
from .unet import UNet
from .unet_residual import UNetResidual

MODELOS = {
    "encoder_decoder": EncoderDecoder,
    "unet": UNet,
    "unet_residual": UNetResidual,
}
