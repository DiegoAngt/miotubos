"""Prepara anotaciones editables y compila solamente imágenes revisadas."""

import argparse
from datetime import datetime
from pathlib import Path

from common.config import RAIZ
from common.yolo_review import compilar_revision, preparar_revision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="accion",required=True)
    preparar = subs.add_parser("prepare",help="copia un lote de vt para corregir polígonos individuales")
    preparar.add_argument("--output",type=Path,default=RAIZ/"data"/f"yolo_review_{datetime.now():%Y%m%d_%H%M%S_%f}")
    preparar.add_argument("--limit",type=int,default=20)
    preparar.add_argument("--seed",type=int,default=42)
    compilar = subs.add_parser("build",help="verifica la revisión y exporta un dataset portable")
    compilar.add_argument("carpeta",type=Path)
    args=parser.parse_args()
    try:
        salida = preparar_revision(args.output,args.limit,args.seed) if args.accion=="prepare" else compilar_revision(args.carpeta)
    except ValueError as error:
        parser.error(str(error))
    print(salida)


if __name__ == "__main__":
    main()
