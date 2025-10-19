from ultralytics import YOLO
import cv2
import numpy as np
from scipy.signal import savgol_filter
import argparse

# --- Argumenty ---
parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, help="Ścieżka do pliku .pt")
parser.add_argument("--video", required=True, help="Ścieżka do pliku wideo")
parser.add_argument("--output", default="output.mp4", help="Plik wynikowy")
parser.add_argument("--rotate", type=int, default=0, help="Obrót klatek: 0, 90, -90")
args = parser.parse_args()

# --- Model YOLO ---
model = YOLO(args.model)

# --- Wideo ---
cap = cv2.VideoCapture(args.video)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Zamień szerokość i wysokość, jeśli obracasz o 90°
if abs(args.rotate) == 90:
    w, h = h, w

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(args.output, fourcc, fps, (w, h))

centers = []

# --- Pętla po klatkach ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Obrót zgodnie z orientacją telefonu
    if args.rotate == 90:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif args.rotate == -90:
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Detekcja YOLO
    results = model(frame, conf=0.25, iou=0.45, verbose=False)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    confs = results[0].boxes.conf.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()

    # Wybierz tylko obiekty klasy 'barbell' (jeśli klasa=0)
    if len(boxes) > 0:
        i = np.argmax(confs)
        x1, y1, x2, y2 = boxes[i]
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        centers.append((cx, cy))

        # Rysuj detekcję
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)

    # --- Rysowanie toru ruchu ---
    if len(centers) > 5:
        xs = np.array([p[0] for p in centers])
        ys = np.array([p[1] for p in centers])

        # Bezpieczny filtr Savitzky-Golay
        if len(xs) >= 7:
            win = min(11, len(xs) - (1 - len(xs) % 2))  # okno nie większe niż liczba próbek
            xs_s = savgol_filter(xs, window_length=win, polyorder=3, mode='interp')
            ys_s = savgol_filter(ys, window_length=win, polyorder=3, mode='interp')
        else:
            xs_s, ys_s = xs, ys

        pts = np.array(list(zip(xs_s.astype(int), ys_s.astype(int))))
        cv2.polylines(frame, [pts], False, (0, 0, 255), 2)

    out.write(frame)

cap.release()
out.release()
print("Zapisano:", args.output)
