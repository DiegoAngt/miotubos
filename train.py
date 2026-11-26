"""Entrena uno de los modelos y guarda todo en results/<model>/.

Uso:
    python train.py encoder_decoder
    python train.py unet
    python train.py unet_residual
"""

import argparse
import random

import numpy as np
import torch

from common.config import RESULTADOS, SEMILLA
from common.data import preparar_division
from common.training import entrenar
from models import MODELOS

parser = argparse.ArgumentParser(description="Entrena un modelo de miotubos.")
parser.add_argument("modelo", choices=MODELOS, help="modelo a entrenar")
nombre = parser.parse_args().modelo

salida = RESULTADOS / nombre
salida.mkdir(parents=True, exist_ok=True)

random.seed(SEMILLA)
np.random.seed(SEMILLA)
torch.manual_seed(SEMILLA)
torch.set_num_threads(2)

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)
print("Modelo:", nombre)
print("Dispositivo:", device)

train, val = preparar_division(salida)
modelo = MODELOS[nombre]().to(device)
entrenar(modelo, nombre, train, val, salida, device)
