"""Rutas y configuración que comparten todos los modelos."""

from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
VT = RAIZ / "vt"
RESULTADOS = RAIZ / "results"

# Configuración inicial para una Mac con 8 GB de memoria.
SEMILLA = 42
AREA_MAXIMA = 0.70
TAMANO = 256
BATCH = 2
EPOCAS = 10
PESO_POSITIVO = 5.0
