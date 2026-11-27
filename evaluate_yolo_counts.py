"""Compara conteos automáticos con conteos manuales revisados, incluyendo bordes."""

import argparse
import json
from pathlib import Path

from common.yolo.evaluation import evaluar, leer_csv


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
