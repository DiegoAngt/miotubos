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
train.py           punto de entrada: elige el modelo y lo entrena
models/            una arquitectura por archivo (blocks.py tiene las piezas comunes)
common/            lo que es igual para todos los modelos
  config.py          rutas, semilla, tamaño de recorte, épocas
  data.py            revisión de etiquetas, división train/val, recortes
  training.py        pérdida, evaluación y ciclo de entrenamiento
exploration/       scripts sueltos: explore.py (ver una imagen) y baseline.py (Otsu)
results/           una carpeta por modelo
vt/                datos (no se sube a git)
```

Cada carpeta de resultados contiene `train.txt`, `val.txt`, `excluded.csv`,
`history.csv`, `diagnostics.csv`, `best_<modelo>.pt` y `last_<modelo>.pt`.

`results/unet_old/` es una corrida de la U-Net con una versión
previa del script; solo conserva los pesos y la división.
