"""Separa y cuenta instancias por recortes; admite CUDA y caché en RAM.

Ejemplo NVIDIA: python predict_yolo.py --device 0 --batch 8 --limit 20
"""

import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import random
import time

import cv2
from PIL import Image

from common.config import RAIZ, RESULTADOS
from common.yolo_counting import exportar_instancias, predecir_instancias
from common.yolo_device import seleccionar_dispositivo


def imagenes_en(origen, limite, semilla):
    extensiones = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    rutas = sorted(p for p in origen.rglob("*") if p.is_file() and p.suffix.lower() in extensiones)
    if limite and len(rutas) > limite:
        rutas = sorted(random.Random(semilla).sample(rutas, limite))
    if not rutas:
        raise ValueError(f"No hay imágenes en {origen}.")
    return rutas


def leer_imagen(ruta):
    imagen = cv2.imread(str(ruta), cv2.IMREAD_COLOR)
    if imagen is None:
        raise ValueError(f"No se pudo leer {ruta}.")
    return imagen


def estimar_ram(rutas):
    total = 0
    for ruta in rutas:
        with Image.open(ruta) as imagen:
            total += imagen.width * imagen.height * 3
    return total


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
    parser.add_argument("--limit", type=int, default=20, help="0 procesa todo; otro valor elige una muestra con semilla fija")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache", choices=["none", "ram"], default="none", help="ram conserva las imágenes seleccionadas descomprimidas")
    parser.add_argument("--estimate-memory", action="store_true", help="estima la caché descomprimida sin cargar modelos ni usar GPU")
    parser.add_argument("--resume", action="store_true", help="continúa una carpeta --output con idéntica configuración")
    args = parser.parse_args()
    if args.batch < 1 or args.limit < 0 or args.tile_size < 32 or args.tile_size % 32:
        parser.error("batch debe ser positivo, limit no negativo y tile-size múltiplo de 32.")
    if not 0 <= args.overlap < 1 or not 0 < args.conf <= 1 or not 0 < args.merge_ios <= 1 or args.min_area < 0:
        parser.error("overlap debe estar en [0,1), conf y merge-ios en (0,1], min-area no negativa.")
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
    import torch
    import ultralytics
    from ultralytics import YOLO

    configuracion = {
        "pipeline_version": 1,
        "source": str(origen), "weights": str(pesos),
        "weights_sha256": hashlib.sha256(pesos.read_bytes()).hexdigest(),
        "tile_size": args.tile_size, "overlap": args.overlap, "conf": args.conf,
        "merge_ios": args.merge_ios, "min_area": args.min_area,
        "seed": args.seed, "limit": args.limit,
        "images": [str(p.relative_to(origen)) for p in rutas],
        "files": [(str(p.relative_to(origen)), p.stat().st_size, p.stat().st_mtime_ns) for p in rutas],
        "ultralytics": ultralytics.__version__, "torch": torch.__version__,
        "criterio": "Contar todos los visibles y marcar los que tocan el borde.",
        "advertencia": "Conteos automáticos provisionales; requieren revisión manual. La fusión puede omitir o unir instancias.",
    }
    ruta_config = destino / "config.json"
    # La configuración de dispositivo y lote no altera la selección de imágenes;
    # se permite reducir el lote al reanudar después de agotar VRAM.
    comparable = json.loads(json.dumps(configuracion))
    if args.resume:
        if not ruta_config.is_file() or json.loads(ruta_config.read_text()) != comparable:
            parser.error("La carpeta no corresponde a los mismos pesos, imágenes o parámetros.")
    else:
        destino.mkdir(parents=True, exist_ok=False)
        ruta_config.write_text(json.dumps(configuracion, indent=2, ensure_ascii=False), encoding="utf-8")
    registro = {"device": device, "batch": args.batch, "cache": args.cache, "fecha": datetime.now().isoformat()}
    (destino / f"ejecucion_{datetime.now():%Y%m%d_%H%M%S_%f}.json").write_text(json.dumps(registro, indent=2))
    print(f"Dispositivo: {device} | Imágenes: {len(rutas)} | Salida: {destino}", flush=True)
    cache = {}
    if args.cache == "ram":
        print(f"Cargando imágenes en RAM; estimación {estimar_ram(rutas) / 1024**3:.2f} GiB más sobrecarga...", flush=True)
        for numero, ruta in enumerate(rutas, 1):
            cache[ruta] = leer_imagen(ruta)
            if numero % 100 == 0:
                print(f"RAM: {numero}/{len(rutas)} imágenes", flush=True)
        print(f"Caché: {sum(im.nbytes for im in cache.values()) / 1024**3:.2f} GiB", flush=True)
    modelo = YOLO(str(pesos))
    if modelo.task != "segment":
        raise ValueError("Se requiere un modelo YOLO de segmentación de instancias.")
    if len(modelo.names) != 1:
        raise ValueError("Se requieren pesos entrenados para la única clase miotubo; los pesos iniciales de COCO no sirven para este conteo.")
    campos = ["imagen", "conteo", "en_borde", "interiores", "recortes", "detecciones_brutas",
              "mascaras_descartadas", "recortes_saturados", "segundos", "estado"]
    filas = []
    with (destino / "conteos.csv").open("w", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=campos)
        escritor.writeheader()
        for numero, ruta in enumerate(rutas, 1):
            relativa = str(ruta.relative_to(origen))
            salida_imagen = destino / "imagenes" / relativa
            resumen = salida_imagen / "resumen.json"
            completos = [salida_imagen / p for p in ("instancias.json", "mascaras.npz", "ids.png", "revision.jpg")]
            if args.resume and resumen.is_file() and all(p.is_file() for p in completos):
                fila = json.loads(resumen.read_text())
            else:
                if salida_imagen.exists():
                    # Una interrupción puede dejar archivos incompletos. Se conservan
                    # en otra carpeta para que el nuevo intento tenga salida atómica.
                    salida_imagen.rename(salida_imagen.with_name(salida_imagen.name + f".incompleta_{time.time_ns()}"))
                imagen = cache[ruta] if ruta in cache else leer_imagen(ruta)
                inicio = time.perf_counter()
                instancias, estadisticas = predecir_instancias(
                    modelo, imagen, device=device, batch=args.batch, tamano=args.tile_size,
                    solapamiento=args.overlap, confianza=args.conf, fusion=args.merge_ios,
                    area_minima=args.min_area)
                exportar_instancias(instancias, imagen, salida_imagen)
                bordes = sum(i["toca_borde"] for i in instancias)
                fila = {"imagen": relativa, "conteo": len(instancias), "en_borde": bordes,
                        "interiores": len(instancias)-bordes, **estadisticas,
                        "segundos": round(time.perf_counter()-inicio, 3), "estado": "pendiente_revision"}
                resumen.write_text(json.dumps(fila, indent=2, ensure_ascii=False), encoding="utf-8")
            escritor.writerow(fila)
            archivo.flush()
            filas.append(fila)
            print(f"[{numero}/{len(rutas)}] {relativa}: {fila['conteo']} instancias, {fila['en_borde']} en borde", flush=True)
    ruta_manual = destino / "conteo_manual.csv"
    anteriores = {}
    if ruta_manual.is_file():
        with ruta_manual.open(newline="", encoding="utf-8") as f:
            anteriores = {r["imagen"]:r for r in csv.DictReader(f)}
    with ruta_manual.open("w", newline="", encoding="utf-8") as f:
        campos_manual = ["imagen", "revisado", "conteo_manual", "borde_manual", "comentarios"]
        escritor = csv.DictWriter(f, fieldnames=campos_manual)
        escritor.writeheader()
        for fila in filas:
            escritor.writerow(anteriores.get(fila["imagen"], {"imagen": fila["imagen"], "revisado": "no"}))
    print("Terminado. Revisa revision.jpg y completa conteo_manual.csv antes de evaluar el conteo.", flush=True)


if __name__ == "__main__":
    main()
