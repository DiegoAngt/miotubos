"""Separa y cuenta instancias por recortes; admite CUDA y caché en RAM.

Ejemplo NVIDIA: python predict_yolo.py --device 0 --batch 8 --limit 20
"""

import argparse
from datetime import datetime
from pathlib import Path

from common.config import RAIZ, RESULTADOS
from common.yolo.device import seleccionar_dispositivo
from common.yolo.prediction import contar_carpeta, estimar_ram, imagenes_en


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=RAIZ / "subset_10GB")
    parser.add_argument("--weights", type=Path, default=RESULTADOS / "yolo_recortes/miotubos_yolo11n_recortes/weights/best.pt")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="auto", help="auto, cpu, mps o índice NVIDIA: 0")
    parser.add_argument("--batch", type=int, default=4, help="recortes simultáneos en la GPU")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--merge-ios", type=float, default=0.5)
    parser.add_argument("--min-area", type=float, default=16)
    parser.add_argument("--max-area", type=float, default=0.25, help="fracción máxima del recorte que puede cubrir una máscara")
    parser.add_argument("--limit", type=int, default=20, help="0 procesa todo; otro valor elige una muestra con semilla fija")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache", choices=["none", "ram"], default="none", help="ram conserva las imágenes seleccionadas descomprimidas")
    parser.add_argument("--estimate-memory", action="store_true", help="estima la caché descomprimida sin cargar modelos ni usar GPU")
    parser.add_argument("--resume", action="store_true", help="continúa una carpeta --output con idéntica configuración")
    args = parser.parse_args()
    if args.batch < 1 or args.limit < 0 or args.tile_size < 32 or args.tile_size % 32:
        parser.error("batch debe ser positivo, limit no negativo y tile-size múltiplo de 32.")
    if not 0 <= args.overlap < 1 or not 0 < args.conf <= 1 or not 0 < args.merge_ios <= 1 or args.min_area < 0 or not 0 < args.max_area <= 1:
        parser.error("overlap debe estar en [0,1), conf, merge-ios y max-area en (0,1], min-area no negativa.")
    if args.resume and args.output is None:
        parser.error("--resume requiere --output.")
    origen = args.source.resolve()
    pesos = args.weights.resolve()
    destino = (args.output or RESULTADOS / "yolo_counts" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")).resolve()
    if destino == origen or origen in destino.parents:
        parser.error("La salida debe quedar fuera de la carpeta de imágenes.")
    rutas = imagenes_en(origen, args.limit, args.seed)
    if args.estimate_memory:
        print(f"{len(rutas)} imágenes: {estimar_ram(rutas) / 1024**3:.2f} GiB de caché BGR, sin modelo ni copias.")
        return
    if not pesos.is_file():
        parser.error(f"No existen los pesos: {pesos}")
    try:
        device = seleccionar_dispositivo(args.device)
    except ValueError as error:
        parser.error(str(error))
    try:
        contar_carpeta(rutas, origen, pesos, destino, device, batch=args.batch,
                       tile_size=args.tile_size, overlap=args.overlap, conf=args.conf,
                       merge_ios=args.merge_ios, min_area=args.min_area, max_area=args.max_area, seed=args.seed,
                       limit=args.limit, cache=args.cache, resume=args.resume)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
