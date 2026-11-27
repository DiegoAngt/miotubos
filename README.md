# Segmentación de miotubos

Tres modelos entrenados con las mismas imágenes de `vt/`, la misma división
train/val y la misma pérdida, para poder compararlos directamente.

| Nombre            | Arquitectura                               | Código                       | Resultados                   |
|-------------------|--------------------------------------------|------------------------------|------------------------------|
| `encoder_decoder` | Encoder-decoder sin conexiones de salto    | `models/encoder_decoder.py` | `results/encoder_decoder/`|
| `unet`            | U-Net simple (con conexiones de salto)     | `models/unet.py`            | `results/unet/`           |
| `unet_residual`   | U-Net con bloques residuales y GroupNorm   | `models/unet_residual.py`   | `results/unet_residual/`  |

## Entrenar

```
python train.py unet_residual
```

El nombre del modelo es el mismo en el comando, en el archivo de `models/` y
en la carpeta de `results/`.

## Estructura

```
train.py           punto de entrada para encoder_decoder, unet y unet_residual
train_yolo.py      punto de entrada para yolo y yolo_recortes
predict_subset.py  predicción por recortes con la U-Net entrenada
models/            una arquitectura por archivo (blocks.py tiene las piezas comunes)
common/            lo que es igual para todos los modelos
  config.py          rutas, semilla, tamaño de recorte, épocas
  data.py            revisión de etiquetas, división train/val, recortes
  training.py        pérdida, evaluación y ciclo de entrenamiento
  yolo_data.py       preparación de imágenes completas y recortes para YOLO
  yolo_training.py   entrenamiento y validación de YOLO
exploration/       scripts sueltos: explore.py (ver una imagen) y baseline.py (Otsu)
results/           una carpeta por modelo
data/              datasets YOLO generados (no se sube a git)
vt/                datos originales (no se sube a git)
subset_10GB/       imágenes para predicción (no se sube a git)
```

Cada carpeta de resultados de los tres modelos anteriores contiene `train.txt`, `val.txt`, `excluded.csv`,
`history.csv`, `diagnostics.csv`, `best_<modelo>.pt` y `last_<modelo>.pt`.

`results/unet_old/` es una corrida de la U-Net con una versión
previa del script; solo conserva los pesos y la división.

## YOLO11n-seg

Dos pruebas de segmentación de instancias conservan las listas `train.txt` y
`val.txt` de `results/unet/`: `yolo` usa imágenes completas a 640 y
`yolo_recortes` usa cuatro recortes nativos de 512 por imagen de entrenamiento
y uno fijo por imagen de validación. Las imágenes fuente no se mezclan entre
entrenamiento y validación.

```bash
python -m pip install ultralytics shapely
python train_yolo.py yolo --epochs 10
python train_yolo.py yolo_recortes --epochs 10
```

Se puede añadir `--device cpu`, `--device mps` o `--solo-preparar`.
Los datasets se generan en `data/yolo/` y `data/yolo_recortes/`, con una carpeta
por ejecución. Los pesos iniciales están en `models/weights/yolo11n-seg.pt`.
Las corridas, pesos entrenados, gráficas y métricas se guardan en
`results/yolo/` y `results/yolo_recortes/`.

Las anotaciones son parciales: el fondo sin marcar se trata como negativo
provisional. El mAP de máscaras de YOLO no equivale al Dice de U-Net; tampoco
se comparan directamente los mAP de imágenes completas y recortes.

## Predicción con U-Net

```bash
python predict_subset.py
```

Usa `models/unet.py` y los pesos `results/unet/best_unet.pt`. Procesa las primeras
cinco imágenes PNG de `subset_10GB/` mediante ventanas de 256 con paso 128 y
promedia las probabilidades en las zonas superpuestas. Guarda máscaras, mapas
de probabilidad y superposiciones en `results/predictions_subset/`, conservando
las subcarpetas de origen.
