# BarPathAI

## Trening modelu YOLO
Trening z pretrenowanego modelu (w tym przypadku `yolov9t.pt`):

```bash
yolo detect train \
  model=yolov9t.pt \
  data=dataset/data.yaml \
  epochs=50 \
  imgsz=640 \
  batch=16 \
```

Wynik: zapisany model w `runs/detect/train/weights/best.pt`

## Testowanie modelu na obrazach / wideo

### 1. Pojedynczy obraz

```bash
yolo detect predict \
  model=runs/detect/train/weights/best.pt \
  source=path/to/image.jpg \
  conf=0.25
```

### 2. Cały folder

```bash
yolo detect predict \
  model=runs/detect/train/weights/best.pt \
  source=dataset/valid/images \
  save=True
```



## Analiza toru ruchu

Uruchamianie analizy z własnym filmem:

```bash
python detect_barbell_path.py \
  --model runs/detect/train/weights/best.pt \
  --video <example_video_path>/test.mp4 \
  --output <example_output_path>/result.mp4 \
  --rotate 90
```

## Zbiór danych

https://universe.roboflow.com/babell-tracking/barbel-tracking-za7nr