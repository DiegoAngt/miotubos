"""Instancias en coordenadas de imagen y fusión de duplicados entre recortes.

La fusión es una heurística de solapamiento de máscaras, no una garantía de
identidad biológica. Conserva separadas las predicciones de un mismo recorte.
"""

from dataclasses import dataclass
import math

import cv2
import numpy as np
from shapely import make_valid
from shapely.geometry import Polygon, mapping
from shapely.ops import unary_union
from shapely.strtree import STRtree


@dataclass
class Deteccion:
    geometria: object
    confianza: float
    clase: int
    recorte: int


def posiciones(longitud, tamano, paso):
    if longitud <= 0 or tamano <= 0 or not 0 < paso <= tamano:
        raise ValueError("Dimensiones o paso de recorte inválidos.")
    ultimo = max(0, longitud - tamano)
    valores = list(range(0, ultimo + 1, paso))
    if valores[-1] != ultimo:
        valores.append(ultimo)
    return valores


def ventanas(ancho, alto, tamano=512, solapamiento=0.5):
    if not 0 <= solapamiento < 1:
        raise ValueError("El solapamiento debe estar entre 0 y 1, sin incluir 1.")
    paso = max(1, round(tamano * (1 - solapamiento)))
    return [(x, y, min(x + tamano, ancho), min(y + tamano, alto))
            for y in posiciones(alto, tamano, paso)
            for x in posiciones(ancho, tamano, paso)]


def solo_poligonos(geometria):
    if geometria.is_empty:
        return []
    if geometria.geom_type == "Polygon":
        return [geometria] if geometria.area > 0 else []
    return [p for g in getattr(geometria, "geoms", []) for p in solo_poligonos(g)]


def geometria_mascara(mascara, x=0, y=0):
    """Conserva componentes y huecos de una instancia, sin unirlos con puentes."""
    contornos, jerarquia = cv2.findContours(
        mascara.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if jerarquia is None:
        return Polygon()
    poligonos = []
    for i, contorno in enumerate(contornos):
        if jerarquia[0, i, 3] != -1 or len(contorno) < 3:
            continue
        exterior = contorno[:, 0, :] + [x, y]
        huecos = []
        hijo = jerarquia[0, i, 2]
        while hijo != -1:
            if len(contornos[hijo]) >= 3:
                huecos.append(contornos[hijo][:, 0, :] + [x, y])
            hijo = jerarquia[0, hijo, 0]
        poligonos.extend(solo_poligonos(make_valid(Polygon(exterior, huecos))))
    return unary_union(poligonos)


def fusionar(detecciones, umbral=0.5):
    """Agrupa por intersección sobre la menor máscara, solo entre recortes.

    Cada grupo admite como máximo una predicción por recorte. Esto evita que
    una unión transitiva colapse dos objetos que el mismo recorte distinguió.
    """
    if not 0 < umbral <= 1:
        raise ValueError("El umbral de fusión debe estar entre 0 y 1.")
    if not detecciones:
        return []
    geometrias = [d.geometria for d in detecciones]
    indice = STRtree(geometrias)
    padres = list(range(len(detecciones)))
    recortes = [{d.recorte} for d in detecciones]

    def raiz(i):
        while padres[i] != i:
            padres[i] = padres[padres[i]]
            i = padres[i]
        return i

    aristas = []
    for i, a in enumerate(detecciones):
        for j in indice.query(a.geometria):
            j = int(j)
            b = detecciones[j]
            if j <= i or a.clase != b.clase or a.recorte == b.recorte:
                continue
            menor = min(a.geometria.area, b.geometria.area)
            if menor <= 0:
                continue
            proporcion = a.geometria.intersection(b.geometria).area / menor
            if proporcion >= umbral:
                aristas.append((proporcion, i, j))
    for _, i, j in sorted(aristas, reverse=True):
        a, b = raiz(i), raiz(j)
        if a != b and recortes[a].isdisjoint(recortes[b]):
            padres[b] = a
            recortes[a].update(recortes[b])
    grupos = {}
    for i, deteccion in enumerate(detecciones):
        grupos.setdefault(raiz(i), []).append(deteccion)
    instancias = []
    for grupo in grupos.values():
        instancias.append({
            "geometria": unary_union([d.geometria for d in grupo]),
            "confianza": max(d.confianza for d in grupo),
            "clase": grupo[0].clase,
            "recortes": sorted(d.recorte for d in grupo),
        })
    return sorted(instancias, key=lambda i: (-i["confianza"], *i["geometria"].bounds))


def predecir_instancias(modelo, imagen, *, device, batch=4, tamano=512,
                       solapamiento=0.5, confianza=0.25, fusion=0.5, area_minima=16):
    """Recibe BGR uint8. Mantiene solo un lote de recortes en la GPU."""
    if batch < 1:
        raise ValueError("batch debe ser positivo.")
    alto, ancho = imagen.shape[:2]
    regiones = ventanas(ancho, alto, tamano, solapamiento)
    detecciones = []
    descartadas = saturados = 0
    for inicio in range(0, len(regiones), batch):
        lote = regiones[inicio:inicio + batch]
        fotos = [imagen[y1:y2, x1:x2].copy() for x1, y1, x2, y2 in lote]
        resultados = modelo.predict(
            fotos, device=device, batch=batch, imgsz=tamano, conf=confianza,
            iou=0.7, max_det=300, retina_masks=True, tile=False,
            verbose=False, save=False, classes=[0],
        )
        for numero, (r, (x1, y1, x2, y2)) in enumerate(zip(resultados, lote), inicio):
            if r.boxes is not None and len(r.boxes) >= 300:
                saturados += 1
            if r.masks is None:
                continue
            for i in range(len(r.masks)):
                mascara = r.masks[i].data[0].cpu().numpy()
                if mascara.shape != (y2-y1, x2-x1):
                    raise ValueError("YOLO devolvió una máscara de tamaño distinto al recorte.")
                g = geometria_mascara(mascara, x1, y1)
                if g.area < area_minima or g.is_empty:
                    descartadas += 1
                    continue
                detecciones.append(Deteccion(g, float(r.boxes.conf[i]), int(r.boxes.cls[i]), numero))
    instancias = fusionar(detecciones, fusion)
    for i, instancia in enumerate(instancias, 1):
        g = instancia["geometria"]
        xmin, ymin, xmax, ymax = g.bounds
        instancia.update(id=i, toca_borde=bool(
            xmin <= 2 or ymin <= 2 or xmax >= ancho-3 or ymax >= alto-3))
    return instancias, {"recortes": len(regiones), "detecciones_brutas": len(detecciones),
                        "mascaras_descartadas": descartadas, "recortes_saturados": saturados}


def mascara_local(geometria, ancho, alto):
    xmin, ymin, xmax, ymax = geometria.bounds
    x1, y1 = max(0, math.floor(xmin)), max(0, math.floor(ymin))
    x2, y2 = min(ancho, math.ceil(xmax)+1), min(alto, math.ceil(ymax)+1)
    mascara = np.zeros((y2-y1, x2-x1), dtype=np.uint8)
    for poligono in solo_poligonos(geometria):
        exterior = np.rint(np.asarray(poligono.exterior.coords) - [x1, y1]).astype(np.int32)
        pieza = np.zeros_like(mascara)
        cv2.fillPoly(pieza, [exterior], 1)
        for anillo in poligono.interiors:
            hueco = np.rint(np.asarray(anillo.coords) - [x1, y1]).astype(np.int32)
            cv2.fillPoly(pieza, [hueco], 0)
        mascara |= pieza
    return mascara, (x1, y1, x2, y2)


def exportar_instancias(instancias, imagen, destino):
    """JSON conserva solapamientos; PNG de IDs muestra la de mayor confianza."""
    import json

    alto, ancho = imagen.shape[:2]
    if len(instancias) > 65535:
        raise ValueError("Demasiadas instancias para el PNG de 16 bits.")
    ids = np.zeros((alto, ancho), dtype=np.uint16)
    vista = imagen.copy()
    registros = []
    mascaras = {}
    for instancia in instancias:
        identificador = instancia["id"]
        g = instancia["geometria"]
        mascara, (x1, y1, x2, y2) = mascara_local(g, ancho, alto)
        zona = ids[y1:y2, x1:x2]
        zona[(mascara > 0) & (zona == 0)] = identificador
        mascaras[f"instancia_{identificador:05d}"] = mascara
        color = tuple(int(v) for v in np.random.default_rng(identificador).integers(60, 255, 3))
        region = vista[y1:y2, x1:x2]
        dentro = mascara.astype(bool)
        region[dentro] = (region[dentro]*0.6 + np.array(color)*0.4).astype(np.uint8)
        for poligono in solo_poligonos(g):
            puntos = np.rint(poligono.exterior.coords).astype(np.int32)
            cv2.polylines(vista, [puntos], True, color, 2)
        punto = g.representative_point()
        texto = f"{identificador}{'*' if instancia['toca_borde'] else ''}"
        ubicacion = (max(0, min(ancho-25, round(punto.x))), max(18, min(alto-4, round(punto.y))))
        cv2.putText(vista, texto, ubicacion, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 4)
        cv2.putText(vista, texto, ubicacion, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        registros.append({k:v for k,v in instancia.items() if k != "geometria"} | {
            "geometria": mapping(g), "bbox": [x1,y1,x2,y2], "area_px": int(mascara.sum())})
    destino.mkdir(parents=True, exist_ok=False)
    (destino / "instancias.json").write_text(json.dumps(registros, ensure_ascii=False), encoding="utf-8")
    mascaras["bbox"] = np.array([r["bbox"] for r in registros], dtype=np.int32).reshape(-1,4)
    np.savez_compressed(destino / "mascaras.npz", **mascaras)
    if not cv2.imwrite(str(destino / "ids.png"), ids):
        raise OSError("No se pudo guardar el mapa de instancias.")
    escala = min(1, 1600/max(alto,ancho))
    if escala < 1:
        vista = cv2.resize(vista, None, fx=escala, fy=escala, interpolation=cv2.INTER_AREA)
    cabecera = np.full((44, vista.shape[1],3), 255, dtype=np.uint8)
    bordes = sum(i["toca_borde"] for i in instancias)
    cv2.putText(cabecera, f"Conteo provisional: {len(instancias)} | Borde (*): {bordes}",
                (8,28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,0), 2)
    if not cv2.imwrite(str(destino / "revision.jpg"), np.vstack([cabecera,vista])):
        raise OSError("No se pudo guardar la vista de revisión.")
    return registros
