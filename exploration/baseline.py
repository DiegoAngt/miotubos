from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter

vt = Path(__file__).resolve().parents[1] / "vt"
primer_ejemplo = "00ab5852-6a55a585-2551-49dd-be42-a9371d0c40ea_A3_04_03_Best.png"


def cargar_gris(ruta):
    with Image.open(ruta) as archivo:
        return archivo.convert("L")


def umbral_otsu(pixeles):
    histograma = np.bincount(pixeles.ravel(), minlength=256).astype(float)
    niveles = np.arange(256)
    peso_izq = histograma.cumsum()[:-1]
    peso_der = histograma.sum() - peso_izq
    suma_total = (histograma * niveles).sum()
    suma_izq = (histograma * niveles).cumsum()[:-1]

    validos = (peso_izq > 0) & (peso_der > 0)
    if not validos.any():
        return int(pixeles.min())

    media_izq = suma_izq[validos] / peso_izq[validos]
    media_der = (suma_total - suma_izq[validos]) / peso_der[validos]
    varianza = peso_izq[validos] * peso_der[validos] * (media_izq - media_der) ** 2
    return int(niveles[:-1][validos][np.argmax(varianza)])


coberturas = []
for ruta in sorted((vt / "masks_png").glob("*.png")):
    if not (vt / "images" / ruta.name).is_file():
        continue
    cobertura = (np.asarray(cargar_gris(ruta)) == 1).mean()
    if cobertura > 0:
        coberturas.append((cobertura, ruta.name))

mediana = np.median([cobertura for cobertura, _ in coberturas])
nombre_mediana = min(coberturas, key=lambda par: abs(par[0] - mediana))[1]
nombre_maximo = max(coberturas)[1]
ejemplos = [primer_ejemplo, nombre_mediana, nombre_maximo]

fig, ejes = plt.subplots(3, 3, figsize=(13, 9))

for fila, nombre in enumerate(ejemplos):
    imagen = cargar_gris(vt / "images" / nombre)
    anotado = np.asarray(cargar_gris(vt / "masks_png" / nombre)) == 1

    suavizada = np.asarray(imagen.filter(ImageFilter.GaussianBlur(radius=1)))
    umbral = umbral_otsu(suavizada)
    prediccion = suavizada <= umbral

    cobertura = 100 * anotado.mean()
    recuperado = 100 * (prediccion & anotado).sum() / anotado.sum()

    ejes[fila, 0].imshow(imagen, cmap="gray")
    ejes[fila, 0].set_title("Original")
    ejes[fila, 1].imshow(anotado, cmap="gray", vmin=0, vmax=1)
    ejes[fila, 1].set_title(f"Anotación parcial ({cobertura:.2f} %)")
    ejes[fila, 2].imshow(prediccion, cmap="gray", vmin=0, vmax=1)
    ejes[fila, 2].set_title(f"Otsu ({100 * prediccion.mean():.2f} % marcado)")

    print(nombre)
    print(f"  Umbral: {umbral} | Píxeles anotados recuperados: {recuperado:.1f} %")

    for eje in ejes[fila]:
        eje.axis("off")

fig.tight_layout()
plt.show()