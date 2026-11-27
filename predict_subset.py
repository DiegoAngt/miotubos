"""Predice imágenes de subset_10GB con la U-Net compartida del proyecto."""

import numpy as np
import torch
from PIL import Image

from common.config import RAIZ, RESULTADOS, TAMANO
from models.unet import UNet

CARPETA_IMAGENES = RAIZ / "subset_10GB"
PESOS = RESULTADOS / "unet" / "best_unet.pt"
SALIDA = RESULTADOS / "predictions_subset"

LIMITE = 5
PASO = 128  # Recortes con superposición.
UMBRAL = 0.50


def posiciones(longitud):
    """Incluye el último recorte para cubrir también los bordes."""
    valores = list(range(0, longitud - TAMANO + 1, PASO))
    ultimo = longitud - TAMANO

    if not valores or valores[-1] != ultimo:
        valores.append(ultimo)

    return valores


def predecir(modelo, imagen, dispositivo):
    alto, ancho = imagen.shape

    # Rellenar solamente si la imagen es menor que un recorte.
    imagen_ampliada = np.pad(
        imagen,
        ((0, max(0, TAMANO - alto)), (0, max(0, TAMANO - ancho))),
        mode="edge",
    )

    alto_ampliado, ancho_ampliado = imagen_ampliada.shape

    suma = np.zeros((alto_ampliado, ancho_ampliado), dtype=np.float32)
    conteo = np.zeros_like(suma)

    filas = posiciones(alto_ampliado)
    columnas = posiciones(ancho_ampliado)

    with torch.inference_mode():
        for y in filas:
            for x in columnas:
                recorte = imagen_ampliada[
                    y:y + TAMANO, x:x + TAMANO
                ]

                entrada = torch.from_numpy(
                    recorte.copy()
                ).unsqueeze(0).unsqueeze(0).to(dispositivo)

                probabilidad = torch.sigmoid(modelo(entrada))
                probabilidad = probabilidad[0, 0].cpu().numpy()

                suma[y:y + TAMANO, x:x + TAMANO] += probabilidad
                conteo[y:y + TAMANO, x:x + TAMANO] += 1

    # Promediar las probabilidades donde coinciden varios recortes.
    return (suma / conteo)[:alto, :ancho]


def main():
    torch.set_num_threads(2)

    if not PESOS.is_file():
        raise FileNotFoundError(
            f"No encontré los pesos del modelo:\n{PESOS}\n"
            "Revisa dónde se guardó results/unet/best_unet.pt."
        )

    imagenes = sorted(
        ruta
        for ruta in CARPETA_IMAGENES.rglob("*")
        if ruta.is_file() and ruta.suffix.lower() == ".png"
    )[:LIMITE]

    if not imagenes:
        raise FileNotFoundError(
            f"No encontré imágenes PNG en {CARPETA_IMAGENES}"
        )

    dispositivo = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Dispositivo: {dispositivo}")

    modelo = UNet().to(dispositivo)

    checkpoint = torch.load(
        PESOS,
        map_location="cpu",
        weights_only=True,
    )
    modelo.load_state_dict(checkpoint["pesos"])
    modelo.eval()

    SALIDA.mkdir(parents=True, exist_ok=True)

    for numero, ruta in enumerate(imagenes, start=1):
        print(
            f"[{numero}/{len(imagenes)}] Procesando: {ruta.name}",
            flush=True,
        )

        with Image.open(ruta) as archivo:
            original = archivo.convert("L")

        imagen = np.asarray(original, dtype=np.float32) / 255.0
        probabilidades = predecir(modelo, imagen, dispositivo)
        mascara = probabilidades >= UMBRAL

        # Mantener la estructura por si hay nombres repetidos.
        carpeta_salida = SALIDA / ruta.relative_to(
            CARPETA_IMAGENES
        ).parent
        carpeta_salida.mkdir(parents=True, exist_ok=True)

        Image.fromarray(
            mascara.astype(np.uint8) * 255
        ).save(carpeta_salida / f"{ruta.stem}_mascara.png")

        # Mapa de probabilidades para inspección visual.
        Image.fromarray(
            (probabilidades * 255).round().astype(np.uint8)
        ).save(carpeta_salida / f"{ruta.stem}_probabilidades.png")

        # Superponer en rojo las regiones predichas.
        superposicion = np.asarray(
            original.convert("RGB"), dtype=np.float32
        ).copy()

        rojo = np.array([255, 0, 0], dtype=np.float32)
        superposicion[mascara] = (
            0.65 * superposicion[mascara] + 0.35 * rojo
        )

        Image.fromarray(
            superposicion.clip(0, 255).astype(np.uint8)
        ).save(carpeta_salida / f"{ruta.stem}_superposicion.png")

        print(f"  Área predicha: {mascara.mean() * 100:.2f}%")

    print(f"\nResultados guardados en:\n{SALIDA}")


if __name__ == "__main__":
    main()
