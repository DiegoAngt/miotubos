"""Compara conteos automáticos con conteos manuales revisados, incluyendo bordes."""

import argparse
import csv
import json
import math
from pathlib import Path


def leer_csv(ruta):
    with ruta.open(newline="", encoding="utf-8-sig") as f:
        filas = list(csv.DictReader(f))
    nombres = [r["imagen"] for r in filas]
    if len(set(nombres)) != len(nombres):
        raise ValueError(f"Hay imágenes duplicadas en {ruta}.")
    return {r["imagen"]:r for r in filas}


def entero(valor, nombre):
    try:
        numero = int(valor)
    except (ValueError, TypeError):
        raise ValueError(f"{nombre} debe ser un entero no negativo.") from None
    if numero < 0:
        raise ValueError(f"{nombre} debe ser un entero no negativo.")
    return numero


def evaluar(predicciones, manuales):
    comparaciones = []
    for imagen, fila in manuales.items():
        if fila.get("revisado", "").strip().lower() not in {"si", "sí", "true", "1"}:
            continue
        if imagen not in predicciones:
            raise ValueError(f"No hay predicción para la imagen revisada {imagen}.")
        real = entero(fila.get("conteo_manual"), f"Conteo manual de {imagen}")
        borde = entero(fila.get("borde_manual"), f"Conteo del borde de {imagen}")
        if borde > real:
            raise ValueError(f"El conteo del borde excede el total en {imagen}.")
        pred = entero(predicciones[imagen]["conteo"], "Conteo automático")
        pred_borde = entero(predicciones[imagen]["en_borde"], "Conteo automático del borde")
        if pred_borde > pred:
            raise ValueError(f"Conteos automáticos incompatibles en {imagen}.")
        comparaciones.append({"imagen": imagen, "manual": real, "predicho": pred,
                              "error": pred-real, "error_borde": pred_borde-borde})
    if not comparaciones:
        raise ValueError("No hay conteos manuales revisados. Completa el CSV y marca revisado=si.")
    n = len(comparaciones)
    return {"imagenes_revisadas": n, "imagenes_predichas": len(predicciones),
            "mae": sum(abs(r["error"]) for r in comparaciones)/n,
            "rmse": math.sqrt(sum(r["error"]**2 for r in comparaciones)/n),
            "sesgo": sum(r["error"] for r in comparaciones)/n,
            "mae_borde": sum(abs(r["error_borde"]) for r in comparaciones)/n,
            "comparaciones": comparaciones,
            "nota": "El error de conteo no mide por sí solo la separación correcta de cada instancia; omisiones y falsos positivos pueden compensarse."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="carpeta generada por predict_yolo.py")
    args = parser.parse_args()
    try:
        resultado = evaluar(leer_csv(args.run/"conteos.csv"), leer_csv(args.run/"conteo_manual.csv"))
    except ValueError as error:
        parser.error(str(error))
    ruta = args.run / "evaluacion_conteo.json"
    ruta.write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(resultado, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
