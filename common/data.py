"""Revisión de etiquetas, división train/val y recortes para entrenar."""

import csv
import random

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from .config import AREA_MAXIMA, SEMILLA, TAMANO, VT


def area_poligono(puntos):
    """Área relativa mediante la fórmula del cordón."""
    x = puntos[:, 0]
    y = puntos[:, 1]
    return 0.5 * abs(
        np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))
    )


def revisar_etiquetas():
    """Devuelve las imágenes aceptadas y las excluidas con su motivo."""
    aceptadas = []
    excluidas = []

    for imagen in sorted((VT / "images").glob("*.png")):
        mascara = VT / "masks_png" / imagen.name
        etiqueta = VT / "labels" / f"{imagen.stem}.txt"

        if not mascara.exists() or not etiqueta.exists():
            excluidas.append((imagen.name, "Falta máscara o etiqueta"))
            continue

        areas = []
        try:
            for linea in etiqueta.read_text().splitlines():
                if not linea.strip():
                    continue

                valores = np.array(linea.split(), dtype=float)
                coordenadas = valores[1:]

                if (
                    len(coordenadas) < 6
                    or len(coordenadas) % 2 != 0
                    or not np.isfinite(valores).all()
                    or (coordenadas < 0).any()
                    or (coordenadas > 1).any()
                    or valores[0] != 0
                ):
                    raise ValueError("Etiqueta inválida")

                areas.append(area_poligono(coordenadas.reshape(-1, 2)))

            if areas and max(areas) > AREA_MAXIMA:
                excluidas.append((imagen.name, "Polígono mayor al 70 %"))
                continue

            with Image.open(imagen) as foto, Image.open(mascara) as mask:
                if foto.size != mask.size:
                    raise ValueError("Dimensiones distintas")
                pixeles = np.asarray(mask.convert("L"))

            if not np.isin(pixeles, [0, 1]).all():
                raise ValueError("Máscara con valores distintos de 0 y 1")

            # Las máscaras vacías tienen supervisión desconocida.
            if not np.any(pixeles == 1):
                excluidas.append((imagen.name, "Máscara sin anotaciones"))
                continue

            aceptadas.append(imagen.name)

        except (ValueError, OSError) as error:
            excluidas.append((imagen.name, str(error)))

    return aceptadas, excluidas


def preparar_division(salida):
    """Crea la división train/val y la guarda en la carpeta de salida.

    Usa el generador global `random`, que debe estar sembrado con SEMILLA
    para que todos los modelos reciban la misma división.
    """
    aceptadas, excluidas = revisar_etiquetas()

    with (salida / "excluded.csv").open("w", newline="") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(["imagen", "motivo"])
        escritor.writerows(excluidas)

    if len(aceptadas) < 4:
        raise RuntimeError("Se necesitan al menos cuatro pares aceptados.")

    # División provisional para comprobar que el entrenamiento funciona.
    # Falta verificar los grupos biológicos para evaluar generalización.
    random.shuffle(aceptadas)
    corte = min(len(aceptadas) - 1, max(1, int(0.80 * len(aceptadas))))
    train = aceptadas[:corte]
    val = aceptadas[corte:]

    for nombre, lista in [("train", train), ("val", val)]:
        (salida / f"{nombre}.txt").write_text("\n".join(lista))

    print("Entrenamiento:", len(train))
    print("Validación provisional:", len(val))
    print("Excluidas:", len(excluidas))
    return train, val


class Miotubos(Dataset):
    """Recortes centrados en anotaciones. Validación fija por imagen."""
    def __init__(self, nombres, entrenamiento=False):
        self.nombres = nombres
        self.entrenamiento = entrenamiento

    def __len__(self):
        return len(self.nombres) * (4 if self.entrenamiento else 1)

    def __getitem__(self, indice):
        nombre = self.nombres[indice % len(self.nombres)]
        with Image.open(VT / "masks_png" / nombre) as archivo:
            mascara = np.array(archivo.convert("L"))
        with Image.open(VT / "images" / nombre) as archivo:
            imagen = np.array(archivo.convert("L"), dtype=np.float32) / 255.0

        alto, ancho = mascara.shape
        # Un generador fijo produce siempre el mismo recorte de validación.
        rng = random if self.entrenamiento else random.Random(SEMILLA + indice)
        filas, columnas = np.nonzero(mascara == 1)
        elegido = rng.randrange(len(filas))
        arriba = max(0, min(int(filas[elegido]) - TAMANO // 2, alto - TAMANO))
        izquierda = max(0, min(int(columnas[elegido]) - TAMANO // 2, ancho - TAMANO))
        x = imagen[arriba:arriba + TAMANO, izquierda:izquierda + TAMANO]
        y = mascara[arriba:arriba + TAMANO, izquierda:izquierda + TAMANO]
        relleno = ((0, TAMANO - x.shape[0]), (0, TAMANO - x.shape[1]))
        validos = np.pad(np.ones_like(y, dtype=np.float32), relleno)
        x = np.pad(x, relleno, mode="edge")
        y = np.pad(y, relleno)
        return (
            torch.from_numpy(x[None].copy()),
            torch.from_numpy(y[None].astype(np.float32)),
            torch.from_numpy(validos[None]),
        )
