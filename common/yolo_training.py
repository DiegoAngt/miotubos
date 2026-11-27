"""Entrenamiento y validación de las dos variantes YOLO11n-seg."""

from pathlib import Path
import json

from .config import RAIZ, RESULTADOS, SEMILLA


def entrenar(yaml, nombre, epochs, device=None):
    tamano = 512 if nombre == "yolo_recortes" else 640
    corrida = "miotubos_yolo11n_recortes" if nombre == "yolo_recortes" else "miotubos_yolo11n"
    import torch
    import ultralytics
    from ultralytics import YOLO

    torch.set_num_threads(2)
    dispositivo = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    print("Dispositivo:", dispositivo, flush=True)
    modelo = YOLO(str(RAIZ / "models" / "weights" / "yolo11n-seg.pt"))
    modelo.train(
        data=str(yaml), epochs=epochs, imgsz=tamano,
        batch=1, device=dispositivo, workers=0, cache=False,
        amp=False, seed=SEMILLA, deterministic=True, patience=10,
        mosaic=0.0, mixup=0.0, copy_paste=0.0,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
        project=str(RESULTADOS / nombre), name=corrida,
        exist_ok=False, plots=True, save=True,
    )
    salida = Path(modelo.trainer.save_dir)
    (salida / "versiones.json").write_text(json.dumps({
        "torch": torch.__version__, "ultralytics": ultralytics.__version__,
        "dataset": str(yaml), "advertencia": "Anotaciones parciales; validación provisional."
    }, indent=2))
    mejor = salida / "weights" / "best.pt"
    if not mejor.is_file():
        raise RuntimeError(f"No se encontró best.pt. Revisa el entrenamiento: {salida}")
    metricas = YOLO(str(mejor)).val(
        data=str(yaml), imgsz=tamano, batch=1, device=dispositivo,
        workers=0, plots=True, project=str(salida), name="validacion_mejor",
    )
    (salida / "metricas_mejor.json").write_text(json.dumps(
        {k: float(v) for k, v in metricas.results_dict.items()}, indent=2
    ))
    print("\nEntrenamiento terminado. Resultados:", salida)
    print("Mejores pesos:", mejor)
    print("mAP de máscaras no equivale al Dice de U-Net.")
