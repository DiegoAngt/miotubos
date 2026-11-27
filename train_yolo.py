"""Entrena YOLO11n-seg en imágenes completas o recortes nativos de 512.

Uso: python train_yolo.py yolo_recortes --epochs 10
Requiere: python -m pip install ultralytics shapely
"""

import argparse

from common.yolo_data import preparar_completas, preparar_recortes
from common.yolo_training import entrenar


def main():
    parser = argparse.ArgumentParser(description="Entrena una variante YOLO de miotubos.")
    parser.add_argument("modelo", choices=["yolo", "yolo_recortes"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", choices=["mps", "cpu"], default=None)
    parser.add_argument("--solo-preparar", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("epochs debe ser positivo")
    preparar = preparar_recortes if args.modelo == "yolo_recortes" else preparar_completas
    yaml = preparar()
    print("Dataset:", yaml, flush=True)
    if not args.solo_preparar:
        entrenar(yaml, args.modelo, args.epochs, args.device)


if __name__ == "__main__":
    main()
