"""Pérdida, evaluación y ciclo de entrenamiento, iguales para todos los modelos."""

import csv

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import AREA_MAXIMA, BATCH, EPOCAS, PESO_POSITIVO, SEMILLA, TAMANO
from .data import Miotubos

UMBRALES_DIAGNOSTICO = [0.05, 0.10, 0.20, 0.30, 0.50]


def crear_perdida(device):
    # Peso positivo moderado para esta prueba con anotaciones parciales.
    # El fondo sin revisar continúa siendo una suposición experimental.
    bce = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([PESO_POSITIVO], device=device),
        reduction="none",
    )

    def perdida(logits, objetivo, validos):
        probabilidades = torch.sigmoid(logits)
        dimensiones = (1, 2, 3)
        interseccion = (probabilidades * objetivo * validos).sum(dimensiones)
        suma = ((probabilidades + objetivo) * validos).sum(dimensiones)
        dice = (2 * interseccion + 1e-6) / (suma + 1e-6)
        error_bce = (bce(logits, objetivo) * validos).sum() / validos.sum()
        return 0.5 * error_bce + 0.5 * (1 - dice.mean())

    return perdida


def evaluar(modelo, cargador, umbrales, device):
    conteos = np.zeros((len(umbrales), 4), dtype=np.int64)
    minimo, maximo = 1.0, 0.0
    positivos = total = 0
    with torch.no_grad():
        for x, y, validos in cargador:
            probabilidades = torch.sigmoid(modelo(x.to(device))).cpu().numpy()
            objetivo = y.numpy() == 1
            region = validos.numpy() == 1
            minimo = min(minimo, float(probabilidades[region].min()))
            maximo = max(maximo, float(probabilidades[region].max()))
            positivos += int((objetivo & region).sum())
            total += int(region.sum())
            for i, umbral in enumerate(umbrales):
                pred = probabilidades >= umbral
                conteos[i] += [
                    int((pred & objetivo & region).sum()),
                    int((pred & ~objetivo & region).sum()),
                    int((~pred & objetivo & region).sum()),
                    int((pred & region).sum()),
                ]
    resultados = []
    for umbral, (tp, fp, fn, marcados) in zip(umbrales, conteos):
        denominador = 2 * tp + fp + fn
        dice = float(2 * tp / denominador) if denominador else 1.0
        resultados.append((umbral, 100 * marcados / total, dice))
    return resultados, minimo, maximo, positivos


def guardar(modelo, ruta, epoca, dice):
    pesos = {k: v.detach().cpu().clone() for k, v in modelo.state_dict().items()}
    torch.save(
        {"pesos": pesos, "tamano": TAMANO, "semilla": SEMILLA,
         "epoca": epoca, "dice_anotaciones_recortes": dice,
         "peso_positivo": PESO_POSITIVO, "area_maxima": AREA_MAXIMA},
        ruta,
    )


def entrenar(modelo, nombre, train, val, salida, device):
    """Entrena el modelo y deja historial, diagnóstico y pesos en `salida`."""
    optimizador = torch.optim.Adam(modelo.parameters(), lr=0.001)
    perdida = crear_perdida(device)

    cargador_train = DataLoader(
        Miotubos(train, entrenamiento=True), batch_size=BATCH,
        shuffle=True, num_workers=0,
    )
    cargador_val = DataLoader(
        Miotubos(val), batch_size=BATCH, shuffle=False, num_workers=0,
    )
    print("Validación en recortes fijos centrados en anotaciones.", flush=True)
    print("El fondo sin anotación se trata como negativo provisional.", flush=True)
    mejor_dice = -1.0
    historial = []

    for epoca in range(EPOCAS):
        modelo.train()
        perdida_total = 0.0
        for x, y, validos in cargador_train:
            x, y, validos = x.to(device), y.to(device), validos.to(device)
            optimizador.zero_grad()
            loss = perdida(modelo(x), y, validos)
            if not torch.isfinite(loss).item():
                raise RuntimeError("La pérdida dejó de ser finita.")
            loss.backward()
            optimizador.step()
            perdida_total += loss.item() * x.size(0)

        modelo.eval()
        resultados, _, _, _ = evaluar(modelo, cargador_val, [0.5], device)
        dice = resultados[0][2]
        promedio = perdida_total / len(cargador_train.dataset)
        historial.append((epoca + 1, promedio, dice))
        print(
            f"Época {epoca + 1}/{EPOCAS} | Loss: {promedio:.4f} | "
            f"Dice en recortes contra anotaciones: {dice:.4f}", flush=True,
        )
        if dice > mejor_dice:
            mejor_dice = dice
            guardar(modelo, salida / f"best_{nombre}.pt", epoca + 1, dice)
        guardar(modelo, salida / f"last_{nombre}.pt", epoca + 1, dice)
        with (salida / "history.csv").open("w", newline="") as archivo:
            escritor = csv.writer(archivo)
            escritor.writerow(
                ["epoca", "loss_train", "dice_val_recortes_anotaciones"]
            )
            escritor.writerows(historial)

    # Diagnóstico una sola vez, con todos los lotes de validación.
    modelo.eval()
    resultados, minimo, maximo, positivos = evaluar(
        modelo, cargador_val, UMBRALES_DIAGNOSTICO, device
    )
    print("\nDiagnóstico del modelo de la última época")
    print(f"Probabilidad mínima: {minimo:.6f}")
    print(f"Probabilidad máxima: {maximo:.6f}")
    print(f"Píxeles anotados en todos los recortes de validación: {positivos}")
    for umbral, area, dice in resultados:
        print(
            f"Umbral {umbral:.2f} | Área predicha: {area:.3f}% | "
            f"Dice contra anotaciones: {dice:.4f}"
        )
    with (salida / "diagnostics.csv").open("w", newline="") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(
            ["umbral", "area_predicha_porcentaje", "dice_anotaciones"]
        )
        escritor.writerows(resultados)
    print("Terminó. Resultados en:", salida)
