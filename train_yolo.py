"""Entrena YOLO11n-seg en imágenes completas o recortes nativos de 512.

Uso: python train_yolo.py yolo_recortes --epochs 10
Requiere: python -m pip install ultralytics shapely
"""

import argparse
from pathlib import Path

from common.yolo.data import preparar_completas, preparar_recortes
from common.yolo.training import entrenar
from common.yolo.review import compilar_revision


def main():
    parser = argparse.ArgumentParser(description="Entrena una variante YOLO de miotubos.")
    parser.add_argument("modelo", choices=["yolo", "yolo_recortes"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", default="auto", help="auto, cpu, mps o GPU NVIDIA: 0")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--cache", choices=["none", "ram", "disk"], default="none")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--weights", type=Path, help="pesos iniciales para una nueva corrida")
    parser.add_argument("--imgsz", type=int, help="tamaño de entrada, múltiplo de 32")
    parser.add_argument("--train-tile", type=int, help="recortes nativos de Ultralytics sobre imágenes completas revisadas")
    parser.add_argument("--tile-overlap", type=float, default=0.5)
    parser.add_argument("--review-dir", type=Path, help="lote con polígonos y particiones revisados; exporta y entrena imágenes completas")
    parser.add_argument("--solo-preparar", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("epochs debe ser positivo")
    if args.batch < 1 or args.workers < 0 or args.patience < 0:
        parser.error("batch debe ser positivo; workers y patience no negativos")
    if args.imgsz is not None and (args.imgsz < 32 or args.imgsz % 32):
        parser.error("imgsz debe ser un múltiplo positivo de 32")
    if args.train_tile is not None and (args.train_tile < 32 or args.train_tile % 32):
        parser.error("train-tile debe ser múltiplo positivo de 32")
    if not 0 <= args.tile_overlap < 1:
        parser.error("tile-overlap debe estar en [0,1)")
    if args.train_tile and not args.review_dir:
        parser.error("train-tile requiere imágenes completas de --review-dir")
    if args.weights is not None and not args.weights.is_file():
        parser.error("No existe el archivo de pesos iniciales")
    if args.review_dir and args.modelo != "yolo":
        parser.error("El dataset revisado contiene campos completos: usa el modelo yolo (por ejemplo --imgsz 1024).")
    preparar = preparar_recortes if args.modelo == "yolo_recortes" else preparar_completas
    try:
        yaml = compilar_revision(args.review_dir) if args.review_dir else preparar()
    except ValueError as error:
        parser.error(str(error))
    print("Dataset:", yaml, flush=True)
    if not args.solo_preparar:
        entrenar(yaml, args.modelo, args.epochs, args.device,
                 weights=args.weights, batch=args.batch, workers=args.workers,
                 cache=False if args.cache == "none" else args.cache,
                 patience=args.patience, amp=args.amp, imgsz=args.imgsz,
                 tile=args.train_tile, tile_overlap=args.tile_overlap)


if __name__ == "__main__":
    main()
