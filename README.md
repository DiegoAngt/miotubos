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
predict_yolo.py    instancias y conteo por imagen con recortes superpuestos
review_yolo.py     preparación y exportación de anotaciones revisadas
evaluate_yolo_counts.py  comparación contra conteos manuales
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

Se puede añadir `--device cpu`, `--device mps`, `--device 0` (NVIDIA) o `--solo-preparar`.
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

## Separar y contar miotubos con YOLO

El criterio es contar todos los miotubos visibles, incluidos los cortados por el
borde de la imagen. Cada instancia que toca el borde queda marcada por separado.
Los conteos de los pesos actuales son provisionales: no hay todavía una evaluación
contra anotaciones completas ni contra conteos manuales.

### Preparar NVIDIA

Instala primero PyTorch y torchvision con CUDA siguiendo el
[selector oficial de PyTorch](https://pytorch.org/get-started/locally/), de acuerdo
con el sistema operativo y el controlador del equipo. Después:

```bash
python -m pip install -r requirements-yolo.txt
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

`--device 0` exige que la primera GPU CUDA esté disponible; un error de CUDA no
se convierte silenciosamente en una ejecución en CPU. Sin especificar dispositivo,
se elige CUDA, MPS o CPU, en ese orden. El tamaño de lote depende de la VRAM de la
tarjeta; `--batch 8` es un punto de partida ajustable, no un requisito de memoria
garantizado. Si falta VRAM, reduce el lote y reanuda la carpeta de salida.

`vt/`, `subset_10GB/` y `data/` son locales y no viajan con Git. Copia al equipo
NVIDIA el subset para predecir y el lote revisado para entrenar. Los pesos existentes
sí están en el repositorio. El dataset compilado no contiene rutas absolutas de Mac.

### Revisar las instancias antes del nuevo entrenamiento

```bash
python review_yolo.py prepare --limit 20 --output data/revision_instancias_20
```

Esto copia 20 imágenes de `vt`, sus polígonos en JSON compatible con LabelMe,
vistas de las anotaciones y `revision.csv`. Las anotaciones originales no se
modifican. El lote inicial combina imágenes de las particiones actuales; esas
particiones todavía no garantizan independencia biológica.

1. Corrige los polígonos de `labels/` con un editor compatible con LabelMe. Cada
   polígono con etiqueta `miotubo` debe representar un individuo. Completa todos
   los miotubos visibles; conserva `group_id` vacío. Una máscara binaria no identifica
   por sí sola los individuos que se tocan. Revisa cruces y ramificaciones con el
   responsable biológico antes de marcar la imagen como completa.
2. En `revision.csv`, asigna el grupo biológico real y las particiones `train`,
   `val` y `test`. Un grupo no puede aparecer en más de una partición. Reserva los
   grupos de prueba antes de entrenar y mantén esas imágenes fuera de los ajustes.
3. Marca `anotaciones_completas=si` únicamente después de corregir cada imagen.
   Se admiten imágenes realmente vacías si fueron revisadas. El exportador excluye
   las pendientes y rechaza grupos compartidos, duplicados entre particiones y
   polígonos inválidos. Un lote de 20 es un piloto de revisión, no una garantía
   de que haya suficientes ejemplos para un modelo confiable.

```bash
python review_yolo.py build data/revision_instancias_20
python train_yolo.py yolo --review-dir data/revision_instancias_20 --epochs 50 --device 0 --batch 8 --workers 4 --cache ram --imgsz 512 --train-tile 512 --tile-overlap 0.5 --patience 10
```

El entrenamiento vuelve a compilar las anotaciones revisadas en una nueva carpeta
y emplea recortes nativos de los campos completos. Mantiene las imágenes fuente
separadas por grupo. AMP se activa por defecto en CUDA y puede desactivarse con
`--no-amp`. Los pesos iniciales se pueden elegir con `--weights`; se trata de una
nueva corrida, no de restaurar el optimizador de una corrida anterior. `test`
queda reservado: el script selecciona los mejores pesos con `val` y no consulta
automáticamente el conjunto de prueba.

El mAP de esa validación usa la combinación nativa de recortes de Ultralytics.
El conteo de `predict_yolo.py` usa la fusión de máscaras descrita abajo. Para medir
su error hay que evaluar este mismo flujo con conteos manuales, no deducirlo del mAP.

El comando anterior necesita la revisión humana. Si ejecutas el entrenamiento
original sin `--review-dir`, sigue usando las anotaciones parciales de `vt` y
sus resultados no certifican la precisión del conteo.

### Generar predicciones y revisar un piloto

Con los pesos actuales, o pasando los del nuevo entrenamiento mediante `--weights`:

```bash
python predict_yolo.py --device 0 --batch 8 --cache ram --limit 20 --output results/yolo_counts/revision_20
```

Elige una muestra reproducible con semilla 42. Procesa recortes nativos de 512 con
50 % de solapamiento. Tras trasladar las máscaras a la imagen completa, agrupa
duplicados de recortes distintos cuando la intersección cubre al menos el 50 %
de la máscara menor (`--merge-ios`). Cada grupo admite como máximo una predicción
por recorte para evitar colapsar dos instancias que ese recorte ya distinguió.
La fusión sigue siendo una heurística: puede unir miotubos distintos o dejar un
miotubo largo fragmentado. Hay que revisar los resultados, especialmente en cruces.

El umbral inicial `--conf 0.25` filtra candidatos de baja confianza. Bajar el umbral
produce más candidatos y puede aumentar falsos positivos; elígelo con la validación
revisada, sin ajustarlo con el conjunto de prueba. Un conteo de cero solo indica
que no quedaron predicciones con esos parámetros y pesos.

La salida incluye:

- `conteos.csv`: total por imagen, cuántos tocan el borde, cuántos son interiores,
  predicciones antes de fusionar, recortes saturados y duración.
- `imagenes/<nombre>/revision.jpg`: máscaras coloreadas con IDs. Un asterisco marca
  una instancia que toca el borde de la imagen completa, no un borde interno del recorte.
- `instancias.json`: geometrías y confianza de cada individuo predicho.
- `mascaras.npz`: máscaras binarias por instancia recortadas a sus cajas `bbox`.
  Preserva objetos superpuestos; se lee sin permitir pickle.
- `ids.png`: mapa de 16 bits para inspección. En un solapamiento muestra la instancia
  de mayor confianza; el JSON y el NPZ conservan las máscaras individuales completas.
- `conteo_manual.csv`: plantilla para anotar el total manual, los individuos del
  borde y `revisado=si`. Los campos manuales se dejan vacíos deliberadamente.
- `config.json`: pesos, hash, parámetros y selección exacta de imágenes para reproducir
  la corrida. La confianza y la fusión son parámetros distintos del modelo.

Los recortes se envían a la GPU por lotes; `--cache ram` mantiene las imágenes en la
RAM del equipo. Las 3,113 imágenes actuales ocupan aproximadamente **43.83 GiB** al
descomprimirse a tres canales de 8 bits, antes del modelo, recortes y demás procesos.
El tamaño de los PNG en disco no equivale al consumo de RAM. `--cache none` permite
leer una imagen cada vez si se prefiere ahorrar memoria.

```bash
python predict_yolo.py --limit 0 --estimate-memory
```

Después de revisar el piloto, el mismo flujo puede recorrer todo el subset:

```bash
python predict_yolo.py --device 0 --batch 8 --cache ram --limit 0 --output results/yolo_counts/subset_completo
```

Una ejecución interrumpida se continúa con los mismos parámetros y `--resume`:

```bash
python predict_yolo.py --device 0 --batch 4 --cache ram --limit 0 --output results/yolo_counts/subset_completo --resume
```

Se puede cambiar el lote o el dispositivo al reanudar; los pesos, imágenes y parámetros
de conteo deben coincidir. Los resultados completos se reutilizan y los incompletos
se conservan en otra carpeta antes de repetirlos. El CSV manual existente se conserva.

### Medir el error del conteo

Revisa las imágenes originales y las vistas con IDs, completa `conteo_manual.csv`
y ejecuta:

```bash
python evaluate_yolo_counts.py results/yolo_counts/revision_20
python -m unittest discover -s tests -v
```

La evaluación calcula MAE, RMSE, sesgo y MAE del borde únicamente para imágenes
marcadas como revisadas. No utiliza anotaciones parciales como si fueran conteos
exhaustivos. Un error de conteo pequeño no demuestra una separación correcta:
omisiones y falsos positivos pueden compensarse, por lo que también hay que
inspeccionar la correspondencia de cada instancia.
