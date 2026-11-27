"""Lotes de anotación por individuo y exportación de un dataset revisado."""

import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import random
import shutil

import cv2
import numpy as np
from PIL import Image
from shapely.geometry import Polygon

from .config import RESULTADOS, VT
from .yolo_data import leer_lista


def preparar_revision(destino, limite=20, semilla=42):
    if limite < 2:
        raise ValueError("El lote debe incluir al menos dos imágenes.")
    seleccion = []
    rng = random.Random(semilla)
    for split, cantidad in [("train", (limite+1)//2), ("val", limite//2)]:
        nombres = leer_lista(RESULTADOS / "unet" / f"{split}.txt")
        seleccion.extend((split,n) for n in rng.sample(nombres, min(cantidad,len(nombres))))
    destino.mkdir(parents=True, exist_ok=False)
    for folder in ("images", "labels", "overlays"):
        (destino / folder).mkdir()
    filas = []
    for split, nombre in seleccion:
        origen = VT / "images" / nombre
        shutil.copy2(origen, destino / "images" / nombre)
        with Image.open(origen) as im:
            ancho, alto = im.size
        formas = []
        vista = cv2.imread(str(origen))
        for i,linea in enumerate((VT/"labels"/f"{Path(nombre).stem}.txt").read_text().splitlines(),1):
            if not linea.strip():
                continue
            valores = [float(v) for v in linea.split()]
            if valores[0] != 0:
                raise ValueError(f"Clase inesperada en {nombre}.")
            puntos = [[x*ancho,y*alto] for x,y in zip(valores[1::2],valores[2::2])]
            formas.append({"label":"miotubo", "points":puntos, "group_id":None,
                           "shape_type":"polygon", "flags":{}})
            cv2.polylines(vista,[np.rint(puntos).astype(np.int32)],True,(0,0,255),3)
            x,y = map(round,puntos[0])
            cv2.putText(vista,str(i),(x,max(20,y)),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)
        anotacion = {"version":"5.5.0", "flags":{}, "shapes":formas,
                     "imagePath":f"../images/{nombre}", "imageData":None,
                     "imageHeight":alto, "imageWidth":ancho}
        (destino/"labels"/f"{Path(nombre).stem}.json").write_text(json.dumps(anotacion,indent=2),encoding="utf-8")
        escala = min(1,1600/max(ancho,alto))
        vista = cv2.resize(vista,None,fx=escala,fy=escala)
        cv2.imwrite(str(destino/"overlays"/f"{Path(nombre).stem}.jpg"),vista)
        filas.append({"imagen":nombre,"split":split,"grupo_biologico":"",
                      "anotaciones_completas":"no","comentarios":""})
    with (destino/"revision.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)
    (destino/"INSTRUCCIONES.txt").write_text(
        "Cada polígono debe describir un miotubo individual visible completo dentro de la imagen.\n"
        "Contar también los miotubos cortados por el borde. No convertir automáticamente componentes de la máscara binaria en individuos.\n"
        "Abrir los JSON de labels/ con un editor compatible con LabelMe y completar o corregir TODOS los miotubos visibles. Etiqueta: miotubo.\n"
        "Cada forma debe ser un polígono independiente, con group_id vacío. Acordar con el responsable biológico cómo resolver cruces y ramificaciones.\n"
        "En revision.csv, asignar grupo_biologico real y split=train, val o test; el mismo grupo no puede aparecer en particiones distintas.\n"
        "El split inicial copia el anterior y NO certifica independencia biológica. Reservar grupos distintos para test antes de entrenar.\n"
        "Marcar anotaciones_completas=si solo después de revisar y corregir la imagen. Se admiten imágenes sin miotubos si fueron revisadas.\n"
        "Las imágenes pendientes se excluyen de la exportación; se necesitan train, val y test con grupos distintos.\n",
        encoding="utf-8")
    return destino


def compilar_revision(carpeta):
    carpeta = carpeta.resolve()
    with (carpeta/"revision.csv").open(newline="",encoding="utf-8-sig") as f:
        todas = list(csv.DictReader(f))
    filas = [r for r in todas if r.get("anotaciones_completas","").strip().lower() in {"si","sí","true","1"}]
    if not filas:
        raise ValueError("No hay imágenes con anotaciones completas revisadas. Completa revision.csv antes de entrenar.")
    nombres = [r["imagen"] for r in filas]
    if len(nombres) != len(set(nombres)):
        raise ValueError("Hay imágenes duplicadas en revision.csv.")
    grupos, contenidos, listos = {}, {}, []
    for fila in filas:
        nombre = fila["imagen"]
        if Path(nombre).name != nombre:
            raise ValueError("revision.csv debe contener nombres de archivo, sin rutas.")
        split, grupo = fila["split"].strip(), fila.get("grupo_biologico","").strip()
        if split not in {"train","val","test"} or not grupo:
            raise ValueError(f"Falta split válido o grupo biológico en {nombre}.")
        if grupo in grupos and grupos[grupo] != split:
            raise ValueError(f"El grupo biológico {grupo} aparece en particiones distintas.")
        grupos[grupo] = split
        imagen = carpeta/"images"/nombre
        digest = hashlib.sha256(imagen.read_bytes()).hexdigest()
        if digest in contenidos and contenidos[digest] != split:
            raise ValueError(f"La imagen {nombre} está duplicada entre particiones.")
        contenidos[digest] = split
        with Image.open(imagen) as im:
            ancho,alto = im.size
        anotacion = json.loads((carpeta/"labels"/f"{Path(nombre).stem}.json").read_text())
        if (anotacion["imageWidth"],anotacion["imageHeight"]) != (ancho,alto):
            raise ValueError(f"Dimensiones incompatibles en las etiquetas de {nombre}.")
        lineas = []
        for forma in anotacion["shapes"]:
            if forma["label"] != "miotubo" or forma["shape_type"] != "polygon" or forma.get("group_id") is not None:
                raise ValueError(f"Se requiere un polígono independiente por miotubo en {nombre}.")
            coords = np.asarray(forma["points"],dtype=float)
            if coords.ndim != 2 or coords.shape[1] != 2 or len(coords) < 3 or not np.isfinite(coords).all():
                raise ValueError(f"Coordenadas inválidas en {nombre}.")
            if (coords < 0).any() or (coords > [ancho,alto]).any():
                raise ValueError(f"Polígono fuera de la imagen {nombre}.")
            poligono = Polygon(coords)
            if not poligono.is_valid or poligono.area <= 0:
                raise ValueError(f"Polígono vacío o con cruces en {nombre}.")
            normalizadas = coords / [ancho,alto]
            lineas.append("0 " + " ".join(f"{v:.12g}" for v in normalizadas.flat))
        listos.append((fila,imagen,lineas,digest))
    if {r[0]["split"].strip() for r in listos} != {"train","val","test"}:
        raise ValueError("Se necesitan imágenes revisadas en train, val y test, con grupos biológicos distintos.")
    if not any(lineas for fila,_,lineas,_ in listos if fila["split"].strip()=="train"):
        raise ValueError("El entrenamiento necesita al menos una instancia anotada.")
    salida = carpeta / f"dataset_{datetime.now():%Y%m%d_%H%M%S_%f}"
    for split in ("train","val","test"):
        (salida/"images"/split).mkdir(parents=True)
        (salida/"labels"/split).mkdir(parents=True)
    for fila,imagen,lineas,_ in listos:
        split = fila["split"].strip()
        shutil.copy2(imagen,salida/"images"/split/imagen.name)
        (salida/"labels"/split/f"{imagen.stem}.txt").write_text("\n".join(lineas)+("\n" if lineas else ""))
    # Sin path absoluto: la carpeta se puede copiar a Windows o Linux con CUDA.
    yaml = salida/"miotubos.yaml"
    yaml.write_text("train: images/train\nval: images/val\ntest: images/test\nnames:\n  0: miotubo\n")
    (salida/"protocolo.json").write_text(json.dumps({
        "anotaciones_completas":True, "criterio_borde":"contar todos los visibles y marcar los del borde",
        "particion_por_grupo_biologico":True,
        "imagenes":[dict(fila,sha256=digest,instancias=len(lineas)) for fila,_,lineas,digest in listos],
    },indent=2,ensure_ascii=False),encoding="utf-8")
    return yaml
