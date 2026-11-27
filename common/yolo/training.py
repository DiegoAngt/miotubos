"""Entrenamiento y validación de las dos variantes YOLO11n-seg."""

from pathlib import Path
import json

from ..config import RAIZ, RESULTADOS, SEMILLA
from .device import seleccionar_dispositivo


def entrenar(yaml, nombre, epochs, device=None, *, weights=None, batch=1,
             workers=0, cache=False, patience=10, amp=None, imgsz=None,
             tile=None, tile_overlap=0.5):
    tamano = imgsz or (512 if nombre == "yolo_recortes" else 640)
    corrida = "miotubos_yolo11n_recortes" if nombre == "yolo_recortes" else "miotubos_yolo11n"
    import torch
    import ultralytics
    from ultralytics import YOLO

    torch.set_num_threads(2)
    dispositivo = seleccionar_dispositivo(device)
    usar_amp = dispositivo.isdigit() if amp is None else amp
    print("Dispositivo:", dispositivo, flush=True)
    modelo = YOLO(str(weights or RAIZ / "models" / "weights" / "yolo11n-seg.pt"))
    recortes_nativos = {"tile": tile, "tile_overlap": tile_overlap} if tile else {}
    modelo.train(
        data=str(yaml), epochs=epochs, imgsz=tamano,
        batch=batch, device=dispositivo, workers=workers, cache=cache,
        amp=usar_amp, seed=SEMILLA, deterministic=True, patience=patience,
        mosaic=0.0, mixup=0.0, copy_paste=0.0,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
        project=str(RESULTADOS / nombre), name=corrida,
        exist_ok=False, plots=True, save=True,
        **recortes_nativos,
    )
    salida = Path(modelo.trainer.save_dir)
    (salida / "versiones.json").write_text(json.dumps({
        "torch": torch.__version__, "ultralytics": ultralytics.__version__,
        "dataset": str(yaml), "device": dispositivo, "batch": batch,
        "amp": usar_amp, "cache": cache, "epochs_solicitadas": epochs,
        "tile": tile, "tile_overlap": tile_overlap,
        "advertencia": "La validez del conteo depende de anotaciones completas y validación independiente."
    }, indent=2))
    mejor = salida / "weights" / "best.pt"
    if not mejor.is_file():
        raise RuntimeError(f"No se encontró best.pt. Revisa el entrenamiento: {salida}")
    metricas = YOLO(str(mejor)).val(
        data=str(yaml), imgsz=tamano, batch=batch, device=dispositivo,
        workers=workers, plots=True, project=str(salida), name="validacion_mejor",
        **recortes_nativos,
    )
    (salida / "metricas_mejor.json").write_text(json.dumps(
        {k: float(v) for k, v in metricas.results_dict.items()}, indent=2
    ))
    print("\nEntrenamiento terminado. Resultados:", salida)
    print("Mejores pesos:", mejor)
    print("mAP de máscaras no equivale al Dice de U-Net.")
