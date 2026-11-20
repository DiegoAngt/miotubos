from pathlib import Path
import csv
import random

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# Configuración inicial para una Mac con 8 GB de memoria.
RAIZ = Path(__file__).resolve().parent
VT = RAIZ / "vt"
SALIDA = RAIZ / "resultados_unet_recortes"
SALIDA.mkdir(exist_ok=True)

SEMILLA = 42
AREA_MAXIMA = 0.70
TAMANO = 256
BATCH = 2
EPOCAS = 10

random.seed(SEMILLA)
np.random.seed(SEMILLA)
torch.manual_seed(SEMILLA)
torch.set_num_threads(2)

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)
print("Dispositivo:", device)


def area_poligono(puntos):
    """Área relativa mediante la fórmula del cordón."""
    x = puntos[:, 0]
    y = puntos[:, 1]
    return 0.5 * abs(
        np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))
    )


# Revisar las etiquetas antes de crear los conjuntos.
aceptadas = []
excluidas = []

for imagen in sorted((VT / "images").glob("*.png")):
    mascara = VT / "masks_png" / imagen.name
    etiqueta = VT / "labels" / f"{imagen.stem}.txt"

    if not mascara.exists() or not etiqueta.exists():
        excluidas.append((imagen.name, "Falta máscara o etiqueta"))
        continue

    areas = []
    try:
        for linea in etiqueta.read_text().splitlines():
            if not linea.strip():
                continue

            valores = np.array(linea.split(), dtype=float)
            coordenadas = valores[1:]

            if (
                len(coordenadas) < 6
                or len(coordenadas) % 2 != 0
                or not np.isfinite(valores).all()
                or (coordenadas < 0).any()
                or (coordenadas > 1).any()
                or valores[0] != 0
            ):
                raise ValueError("Etiqueta inválida")

            areas.append(area_poligono(coordenadas.reshape(-1, 2)))

        if areas and max(areas) > AREA_MAXIMA:
            excluidas.append((imagen.name, "Polígono mayor al 70 %"))
            continue

        with Image.open(imagen) as foto, Image.open(mascara) as mask:
            if foto.size != mask.size:
                raise ValueError("Dimensiones distintas")
            pixeles = np.asarray(mask.convert("L"))

        if not np.isin(pixeles, [0, 1]).all():
            raise ValueError("Máscara con valores distintos de 0 y 1")

        # Las máscaras vacías tienen supervisión desconocida.
        if not np.any(pixeles == 1):
            excluidas.append((imagen.name, "Máscara sin anotaciones"))
            continue

        aceptadas.append(imagen.name)

    except (ValueError, OSError) as error:
        excluidas.append((imagen.name, str(error)))

with (SALIDA / "excluidas.csv").open("w", newline="") as archivo:
    escritor = csv.writer(archivo)
    escritor.writerow(["imagen", "motivo"])
    escritor.writerows(excluidas)

if len(aceptadas) < 4:
    raise RuntimeError("Se necesitan al menos cuatro pares aceptados.")

# División provisional para comprobar que el entrenamiento funciona.
# Falta verificar los grupos biológicos para evaluar generalización.
random.shuffle(aceptadas)
corte = min(len(aceptadas) - 1, max(1, int(0.80 * len(aceptadas))))
train = aceptadas[:corte]
val = aceptadas[corte:]

for nombre, lista in [("train", train), ("val", val)]:
    (SALIDA / f"{nombre}.txt").write_text("\n".join(lista))

print("Entrenamiento:", len(train))
print("Validación provisional:", len(val))
print("Excluidas:", len(excluidas))


class Miotubos(Dataset):
    """Recortes centrados en anotaciones. Validación fija por imagen."""
    def __init__(self, nombres, entrenamiento=False):
        self.nombres = nombres
        self.entrenamiento = entrenamiento

    def __len__(self):
        return len(self.nombres) * (4 if self.entrenamiento else 1)

    def __getitem__(self, indice):
        nombre = self.nombres[indice % len(self.nombres)]
        with Image.open(VT / "masks_png" / nombre) as archivo:
            mascara = np.array(archivo.convert("L"))
        with Image.open(VT / "images" / nombre) as archivo:
            imagen = np.array(archivo.convert("L"), dtype=np.float32) / 255.0

        alto, ancho = mascara.shape
        # Un generador fijo produce siempre el mismo recorte de validación.
        rng = random if self.entrenamiento else random.Random(SEMILLA + indice)
        filas, columnas = np.nonzero(mascara == 1)
        elegido = rng.randrange(len(filas))
        arriba = max(0, min(int(filas[elegido]) - TAMANO // 2, alto - TAMANO))
        izquierda = max(0, min(int(columnas[elegido]) - TAMANO // 2, ancho - TAMANO))
        x = imagen[arriba:arriba + TAMANO, izquierda:izquierda + TAMANO]
        y = mascara[arriba:arriba + TAMANO, izquierda:izquierda + TAMANO]
        relleno = ((0, TAMANO - x.shape[0]), (0, TAMANO - x.shape[1]))
        validos = np.pad(np.ones_like(y, dtype=np.float32), relleno)
        x = np.pad(x, relleno, mode="edge")
        y = np.pad(y, relleno)
        return (
            torch.from_numpy(x[None].copy()),
            torch.from_numpy(y[None].astype(np.float32)),
            torch.from_numpy(validos[None]),
        )


def bloque(entrada, salida):
    return nn.Sequential(
        nn.Conv2d(entrada, salida, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(salida, salida, 3, padding=1),
        nn.ReLU(),
    )


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = bloque(1, 16)
        self.enc2 = bloque(16, 32)
        self.centro = bloque(32, 64)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = bloque(64, 32)
        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec1 = bloque(32, 16)
        self.salida = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        centro = self.centro(self.pool(e2))
        d2 = self.dec2(torch.cat([self.up2(centro), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.salida(d1)


modelo = UNet().to(device)
optimizador = torch.optim.Adam(modelo.parameters(), lr=0.001)
# Peso positivo moderado para esta prueba con anotaciones parciales.
# El fondo sin revisar continúa siendo una suposición experimental.
bce = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor([5.0], device=device), reduction="none"
)


def perdida(logits, objetivo, validos):
    probabilidades = torch.sigmoid(logits)
    dimensiones = (1, 2, 3)
    interseccion = (probabilidades * objetivo * validos).sum(dimensiones)
    suma = ((probabilidades + objetivo) * validos).sum(dimensiones)
    dice = (2 * interseccion + 1e-6) / (suma + 1e-6)
    error_bce = (bce(logits, objetivo) * validos).sum() / validos.sum()
    return 0.5 * error_bce + 0.5 * (1 - dice.mean())


def evaluar(modelo, cargador, umbrales):
    conteos = np.zeros((len(umbrales), 4), dtype=np.int64)
    minimo, maximo = 1.0, 0.0
    positivos = total = 0
    with torch.no_grad():
        for x, y, validos in cargador:
            probabilidades = torch.sigmoid(modelo(x.to(device))).cpu().numpy()
            objetivo = y.numpy() == 1
            region = validos.numpy() == 1
            minimo = min(minimo, float(probabilidades[region].min()))
            maximo = max(maximo, float(probabilidades[region].max()))
            positivos += int((objetivo & region).sum())
            total += int(region.sum())
            for i, umbral in enumerate(umbrales):
                pred = probabilidades >= umbral
                conteos[i] += [
                    int((pred & objetivo & region).sum()),
                    int((pred & ~objetivo & region).sum()),
                    int((~pred & objetivo & region).sum()),
                    int((pred & region).sum()),
                ]
    resultados = []
    for umbral, (tp, fp, fn, marcados) in zip(umbrales, conteos):
        denominador = 2 * tp + fp + fn
        dice = float(2 * tp / denominador) if denominador else 1.0
        resultados.append((umbral, 100 * marcados / total, dice))
    return resultados, minimo, maximo, positivos


def guardar(nombre, epoca, dice):
    pesos = {k: v.detach().cpu().clone() for k, v in modelo.state_dict().items()}
    torch.save(
        {"pesos": pesos, "tamano": TAMANO, "semilla": SEMILLA,
         "epoca": epoca, "dice_anotaciones_recortes": dice,
         "peso_positivo": 5.0, "area_maxima": AREA_MAXIMA},
        SALIDA / nombre,
    )


cargador_train = DataLoader(
    Miotubos(train, entrenamiento=True), batch_size=BATCH,
    shuffle=True, num_workers=0,
)
cargador_val = DataLoader(
    Miotubos(val), batch_size=BATCH, shuffle=False, num_workers=0,
)
print("Validación en recortes fijos centrados en anotaciones.", flush=True)
print("El fondo sin anotación se trata como negativo provisional.", flush=True)
mejor_dice = -1.0
historial = []

for epoca in range(EPOCAS):
    modelo.train()
    perdida_total = 0.0
    for x, y, validos in cargador_train:
        x, y, validos = x.to(device), y.to(device), validos.to(device)
        optimizador.zero_grad()
        loss = perdida(modelo(x), y, validos)
        if not torch.isfinite(loss).item():
            raise RuntimeError("La pérdida dejó de ser finita.")
        loss.backward()
        optimizador.step()
        perdida_total += loss.item() * x.size(0)

    modelo.eval()
    resultados, _, _, _ = evaluar(modelo, cargador_val, [0.5])
    dice = resultados[0][2]
    promedio = perdida_total / len(cargador_train.dataset)
    historial.append((epoca + 1, promedio, dice))
    print(
        f"Época {epoca + 1}/{EPOCAS} | Loss: {promedio:.4f} | "
        f"Dice en recortes contra anotaciones: {dice:.4f}", flush=True,
    )
    if dice > mejor_dice:
        mejor_dice = dice
        guardar("mejor_unet.pt", epoca + 1, dice)
    guardar("ultima_unet.pt", epoca + 1, dice)
    with (SALIDA / "historial.csv").open("w", newline="") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(["epoca", "loss_train", "dice_val_recortes_anotaciones"])
        escritor.writerows(historial)

# Diagnóstico una sola vez, con todos los lotes de validación.
modelo.eval()
resultados, minimo, maximo, positivos = evaluar(
    modelo, cargador_val, [0.05, 0.10, 0.20, 0.30, 0.50]
)
print("\nDiagnóstico del modelo de la última época")
print(f"Probabilidad mínima: {minimo:.6f}")
print(f"Probabilidad máxima: {maximo:.6f}")
print(f"Píxeles anotados en todos los recortes de validación: {positivos}")
for umbral, area, dice in resultados:
    print(
        f"Umbral {umbral:.2f} | Área predicha: {area:.3f}% | "
        f"Dice contra anotaciones: {dice:.4f}"
    )
with (SALIDA / "diagnostico.csv").open("w", newline="") as archivo:
    escritor = csv.writer(archivo)
    escritor.writerow(["umbral", "area_predicha_porcentaje", "dice_anotaciones"])
    escritor.writerows(resultados)
print("Terminó. Resultados en:", SALIDA)
