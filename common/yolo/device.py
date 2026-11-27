"""Selección explícita de CUDA, MPS o CPU para entrenamiento e inferencia."""


def seleccionar_dispositivo(device=None):
    import torch

    if device in (None, "auto"):
        if torch.cuda.is_available():
            return "0"
        return "mps" if torch.backends.mps.is_available() else "cpu"
    device = str(device).lower()
    if device.startswith("cuda:"):
        device = device.split(":", 1)[1]
    if device == "cuda":
        device = "0"
    if device.isdigit():
        if not torch.cuda.is_available() or int(device) >= torch.cuda.device_count():
            raise ValueError(f"La GPU CUDA {device} no está disponible. Revisa PyTorch y el controlador NVIDIA.")
        return device
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS no está disponible en este equipo.")
    if device not in ("cpu", "mps"):
        raise ValueError("device debe ser auto, cpu, mps o el índice de una GPU CUDA, por ejemplo 0.")
    return device
