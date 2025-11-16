"""Prosty skrypt treningowy z ustalonymi parametrami.

Ten skrypt używa wartości zakodowanych na stałe (brak CLI). Preferuje CUDA, gdy jest
dostępne i w przeciwnym razie używa CPU. Aby zmienić ustawienia treningu, edytuj poniższe stałe.
"""

import sys
import torch
from ultralytics import YOLO


WEIGHTS = "yolov9t.pt"
DATA = "dataset/data.yaml"
EPOCHS = 50
IMGSZ = 640
BATCH = 16

augmentations = {
    "hsv_h": 0.5,     # Zmiana odcienia
    "hsv_s": 0.5,     # Zmiana nasycenia
    "hsv_v": 0.5,     # Zmiana jasności
    "translate": 0.3, # Translacja
    "scale": 0.5,     # Skalowanie
    "shear": 0.3,     # Ścinanie
}


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main():
    device = get_device()
    print(f"Urządzenie ustawione na: {device}")
    print(
        f"Rozpoczęcie treningu z wagami={WEIGHTS}, danymi={DATA}, epokami={EPOCHS}, rozmiarem obrazów={IMGSZ}"
    )

    try:
        model = YOLO(WEIGHTS)
    except Exception as e:
        print(f"Nie udało się załadować wag modelu '{WEIGHTS}': {e}")
        sys.exit(1)

    model.info()

    try:
        model.train(data=DATA, epochs=EPOCHS, imgsz=IMGSZ, device=device, batch=BATCH, patience=5, **augmentations)
    except Exception as e:
        print(f"Trening nie powiódł się: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
