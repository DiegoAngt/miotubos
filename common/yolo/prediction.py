"""Separa y cuenta instancias por recortes sobre una carpeta de imágenes."""

import csv
from datetime import datetime
import hashlib
import json
import random
import time

import cv2
from PIL import Image

from .counting import exportar_instancias, predecir_instancias


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


def contar_carpeta(rutas, origen, pesos, destino, device, *, batch=4, tile_size=512,
                   overlap=0.5, conf=0.25, merge_ios=0.5, min_area=16, max_area=0.25, seed=42,
                   limit=20, cache="none", resume=False):
    import torch
    import ultralytics
    from ultralytics import YOLO

    configuracion = {
        "pipeline_version": 2,
        "source": str(origen), "weights": str(pesos),
        "weights_sha256": hashlib.sha256(pesos.read_bytes()).hexdigest(),
        "tile_size": tile_size, "overlap": overlap, "conf": conf,
        "merge_ios": merge_ios, "min_area": min_area, "max_area": max_area,
        "seed": seed, "limit": limit,
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
    if resume:
        if not ruta_config.is_file() or json.loads(ruta_config.read_text()) != comparable:
            raise ValueError("La carpeta no corresponde a los mismos pesos, imágenes o parámetros.")
    else:
        destino.mkdir(parents=True, exist_ok=False)
        ruta_config.write_text(json.dumps(configuracion, indent=2, ensure_ascii=False), encoding="utf-8")
    registro = {"device": device, "batch": batch, "cache": cache, "fecha": datetime.now().isoformat()}
    (destino / f"ejecucion_{datetime.now():%Y%m%d_%H%M%S_%f}.json").write_text(json.dumps(registro, indent=2))
    print(f"Dispositivo: {device} | Imágenes: {len(rutas)} | Salida: {destino}", flush=True)
    imagenes = {}
    if cache == "ram":
        print(f"Cargando imágenes en RAM; estimación {estimar_ram(rutas) / 1024**3:.2f} GiB más sobrecarga...", flush=True)
        for numero, ruta in enumerate(rutas, 1):
            imagenes[ruta] = leer_imagen(ruta)
            if numero % 100 == 0:
                print(f"RAM: {numero}/{len(rutas)} imágenes", flush=True)
        print(f"Caché: {sum(im.nbytes for im in imagenes.values()) / 1024**3:.2f} GiB", flush=True)
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
            if resume and resumen.is_file() and all(p.is_file() for p in completos):
                fila = json.loads(resumen.read_text())
            else:
                if salida_imagen.exists():
                    # Una interrupción puede dejar archivos incompletos. Se conservan
                    # en otra carpeta para que el nuevo intento tenga salida atómica.
                    salida_imagen.rename(salida_imagen.with_name(salida_imagen.name + f".incompleta_{time.time_ns()}"))
                imagen = imagenes[ruta] if ruta in imagenes else leer_imagen(ruta)
                inicio = time.perf_counter()
                instancias, estadisticas = predecir_instancias(
                    modelo, imagen, device=device, batch=batch, tamano=tile_size,
                    solapamiento=overlap, confianza=conf, fusion=merge_ios,
                    area_minima=min_area, area_maxima=max_area)
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

