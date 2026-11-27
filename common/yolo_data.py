"""Preparación de datasets YOLO con la misma partición que U-Net."""

from datetime import datetime
from pathlib import Path
import csv
import hashlib
import json
import math
import random
import shutil

import numpy as np
from PIL import Image

from .config import AREA_MAXIMA, RAIZ, RESULTADOS, SEMILLA, VT


def leer_lista(ruta):
    if not ruta.is_file():
        raise FileNotFoundError(f"Falta {ruta}. Conserva las listas de U-Net.")
    nombres = [s.strip() for s in ruta.read_text().splitlines() if s.strip()]
    if not nombres or len(nombres) != len(set(nombres)):
        raise ValueError(f"Lista vacía o con duplicados: {ruta}")
    if any(Path(n).name != n for n in nombres):
        raise ValueError(f"Se esperaban nombres de archivo en {ruta}")
    return nombres


def revisar_etiqueta(ruta):
    lineas = []
    for numero, linea in enumerate(ruta.read_text().splitlines(), 1):
        if not linea.strip():
            continue
        valores = [float(s) for s in linea.split()]
        coords = valores[1:]
        if (valores[0] != 0 or len(coords) < 6 or len(coords) % 2
                or not all(math.isfinite(v) for v in valores)
                or any(v < 0 or v > 1 for v in coords)):
            raise ValueError(f"Etiqueta inválida: {ruta}, línea {numero}")
        puntos = list(zip(coords[::2], coords[1::2]))
        area = abs(sum(x * puntos[(i+1) % len(puntos)][1]
                       - y * puntos[(i+1) % len(puntos)][0]
                       for i, (x, y) in enumerate(puntos))) / 2
        if area <= 0 or area > AREA_MAXIMA:
            raise ValueError(f"Polígono vacío o mayor al 70%: {ruta}")
        lineas.append("0 " + " ".join(f"{v:.12g}" for v in coords))
    if not lineas:
        raise ValueError(f"Sin polígonos: {ruta}. No se cambia la partición.")
    return "\n".join(lineas) + "\n"


def preparar_completas():
    particion = RESULTADOS / "unet"
    listas = {k: leer_lista(particion / f"{k}.txt") for k in ("train", "val")}
    if set(listas["train"]) & set(listas["val"]):
        raise ValueError("Hay imágenes compartidas entre entrenamiento y validación.")
    etiquetas = {}
    for grupo, nombres in listas.items():
        tallos = [Path(n).stem for n in nombres]
        if len(tallos) != len(set(tallos)):
            raise ValueError("Nombres de etiqueta repetidos.")
        for nombre in nombres:
            imagen = VT / "images" / nombre
            txt = VT / "labels" / f"{Path(nombre).stem}.txt"
            if not imagen.is_file() or not txt.is_file():
                raise FileNotFoundError(f"Falta imagen o etiqueta: {nombre}")
            etiquetas[nombre] = revisar_etiqueta(txt)

    # Una carpeta nueva por ejecución; no borra ni cambia los originales.
    destino = RAIZ / "data" / "yolo" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    for grupo, nombres in listas.items():
        fotos = destino / "images" / grupo
        labels = destino / "labels" / grupo
        fotos.mkdir(parents=True)
        labels.mkdir(parents=True)
        for nombre in nombres:
            origen = VT / "images" / nombre
            # Enlaces para ahorrar espacio. Copia solo si el sistema no permite enlaces.
            try:
                (fotos / nombre).symlink_to(origen.resolve())
            except OSError:
                shutil.copy2(origen, fotos / nombre)
            (labels / f"{Path(nombre).stem}.txt").write_text(etiquetas[nombre])
        shutil.copy2(particion / f"{grupo}.txt", destino / f"{grupo}.txt")
    yaml = destino / "miotubos.yaml"
    yaml.write_text(
        f"path: {json.dumps(str(destino))}\n"
        "train: images/train\nval: images/val\nnames:\n  0: miotubo\n"
    )
    print(f"Entrenamiento: {len(listas['train'])}")
    print(f"Validación: {len(listas['val'])}")
    print("Mismas imágenes que U-Net; esta prueba usa imágenes completas a 640.")
    print("Anotaciones parciales: los objetos sin marcar pueden penalizarse como fondo.")
    return yaml


TAMANO = 512


def fragmentos(geometria):
    if geometria.is_empty:
        return []
    if geometria.geom_type == "Polygon":
        return [geometria] if geometria.area > 0 else []
    if hasattr(geometria, "geoms"):
        return [p for g in geometria.geoms for p in fragmentos(g)]
    return []


def etiquetas_recorte(poligonos, izquierda, arriba):
    from shapely.geometry import box

    ventana = box(izquierda, arriba, izquierda + TAMANO, arriba + TAMANO)
    lineas = []
    for poligono in poligonos:
        partes = fragmentos(poligono.intersection(ventana))
        # Un objeto cortado en dos componentes desconectados no se convierte
        # artificialmente en dos instancias: probar otra posición del recorte.
        if len(partes) > 1:
            return None
        for parte in partes:
            coords = []
            for x, y in list(parte.exterior.coords)[:-1]:
                coords.extend([max(0, min(1, (x - izquierda) / TAMANO)),
                               max(0, min(1, (y - arriba) / TAMANO))])
            lineas.append("0 " + " ".join(f"{v:.12g}" for v in coords))
    return lineas


def preparar_recortes():
    from shapely.geometry import Polygon

    particion = RESULTADOS / "unet"
    listas = {k: leer_lista(particion / f"{k}.txt") for k in ("train", "val")}
    if set(listas["train"]) & set(listas["val"]):
        raise ValueError("Hay imágenes compartidas entre entrenamiento y validación.")
    # Validar primero nombres y etiquetas; no cambiar silenciosamente la partición.
    textos = {}
    for nombres in listas.values():
        for nombre in nombres:
            imagen = VT / "images" / nombre
            txt = VT / "labels" / f"{Path(nombre).stem}.txt"
            mascara = VT / "masks_png" / nombre
            if not all(p.is_file() for p in (imagen, txt, mascara)):
                raise FileNotFoundError(f"Falta imagen, máscara o TXT: {nombre}")
            textos[nombre] = revisar_etiqueta(txt)

    destino = RAIZ / "data" / "yolo_recortes" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    destino.mkdir(parents=True)
    registros = []
    for grupo, nombres in listas.items():
        fotos = destino / "images" / grupo
        labels = destino / "labels" / grupo
        fotos.mkdir(parents=True)
        labels.mkdir(parents=True)
        cantidad = 4 if grupo == "train" else 1
        for indice, nombre in enumerate(nombres):
            with Image.open(VT / "images" / nombre) as archivo:
                foto = archivo.convert("L")
            with Image.open(VT / "masks_png" / nombre) as archivo:
                mascara = np.array(archivo.convert("L"))
            ancho, alto = foto.size
            if mascara.shape != (alto, ancho) or min(ancho, alto) < TAMANO:
                raise ValueError(f"Dimensiones incompatibles con recortes 512: {nombre}")
            if not np.isin(mascara, [0, 1]).all():
                raise ValueError(f"La máscara no es 0/1: {nombre}")
            filas, columnas = np.nonzero(mascara == 1)
            if not len(filas):
                raise ValueError(f"Máscara vacía dentro de la partición: {nombre}")
            poligonos = []
            for linea in textos[nombre].splitlines():
                coords = [float(v) for v in linea.split()[1:]]
                poligono = Polygon([(x*ancho, y*alto)
                                     for x, y in zip(coords[::2], coords[1::2])])
                if not poligono.is_valid:
                    raise ValueError(f"Polígono con geometría inválida: {nombre}")
                poligonos.append(poligono)
            # El recorte de val es fijo; el entrenamiento usa cuatro posiciones
            # fijas por imagen. No se mezclan imágenes fuente entre conjuntos.
            rng = random.Random(SEMILLA + indice + (100000 if grupo == "train" else 0))
            for numero in range(cantidad):
                for intento in range(100):
                    elegido = rng.randrange(len(filas))
                    arriba = max(0, min(int(filas[elegido])-TAMANO//2, alto-TAMANO))
                    izquierda = max(0, min(int(columnas[elegido])-TAMANO//2, ancho-TAMANO))
                    lineas = etiquetas_recorte(poligonos, izquierda, arriba)
                    if lineas:
                        break
                else:
                    raise ValueError(f"No se pudo crear recorte válido para {nombre}")
                identificador = f"{indice:04d}_{numero:02d}"
                foto.crop((izquierda, arriba, izquierda+TAMANO, arriba+TAMANO)).convert("RGB").save(
                    fotos / f"{identificador}.png")
                (labels / f"{identificador}.txt").write_text("\n".join(lineas)+"\n")
                registros.append([grupo, nombre, identificador, izquierda, arriba,
                                  TAMANO, len(lineas)])
        shutil.copy2(particion / f"{grupo}.txt", destino / f"fuentes_{grupo}.txt")
        print(f"{grupo}: {len(nombres)} imágenes fuente, {len(nombres)*cantidad} recortes", flush=True)
    with (destino / "recortes.csv").open("w", newline="") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(["grupo", "imagen_fuente", "recorte", "x", "y", "tamano", "poligonos_visibles"])
        escritor.writerows(registros)
    yaml = destino / "miotubos.yaml"
    yaml.write_text(f"path: {json.dumps(str(destino))}\n"
                    "train: images/train\nval: images/val\nnames:\n  0: miotubo\n")
    (destino / "protocolo.json").write_text(json.dumps({
        "semilla": SEMILLA, "tamano": TAMANO, "recortes_train_por_imagen": 4,
        "recortes_val_por_imagen": 1, "seleccion": "centrados en píxeles anotados",
        "advertencia": "Validación condicionada a anotaciones parciales; no mide conteo exhaustivo ni desempeño en campo completo.",
        "limite_area_original": AREA_MAXIMA,
        "hash_listas": {k: hashlib.sha256((particion/f"{k}.txt").read_bytes()).hexdigest()
                        for k in listas}
    }, ensure_ascii=False, indent=2))
    print("Anotaciones parciales: fondo no anotado se trata como negativo provisional.")
    print("No comparar directamente este mAP con mAP de imágenes completas ni Dice de U-Net.")
    return yaml
