from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

vt = Path(__file__).resolve().parents[1] / "vt"
nombre = "00ab5852-6a55a585-2551-49dd-be42-a9371d0c40ea_A3_04_03_Best.png"

with Image.open(vt / "images" / nombre) as archivo:
    imagen = np.array(archivo.convert("L"))

with Image.open(vt / "masks_png" / nombre) as archivo:
    mascara = np.array(archivo.convert("L"))

if imagen.shape != mascara.shape:
    raise ValueError(f"Tamaños distintos: {imagen.shape} y {mascara.shape}")

marcado = mascara == 1
print(f"Tamaño: {imagen.shape[1]} × {imagen.shape[0]}")
print(f"Valores de la máscara: {np.unique(mascara)}")
print(f"Porcentaje anotado: {100 * marcado.mean():.3f}%")

fig, ejes = plt.subplots(1, 3, figsize=(14, 5))

ejes[0].imshow(imagen, cmap="gray")
ejes[0].set_title("Imagen original")

ejes[1].imshow(mascara, cmap="gray", vmin=0, vmax=1)
ejes[1].set_title("Máscara anotada")

ejes[2].imshow(imagen, cmap="gray")
ejes[2].contour(marcado, levels=[0.5], colors="red", linewidths=1)
ejes[2].set_title("Contornos sobre la imagen")

for eje in ejes:
    eje.axis("off")

fig.tight_layout()
plt.show()