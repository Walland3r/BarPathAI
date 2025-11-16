from ultralytics import YOLO
import cv2
import numpy as np
from scipy.signal import savgol_filter
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--video", required=True)
parser.add_argument("--output", default="output.mp4")
parser.add_argument("--rotate", type=int, default=0)
args = parser.parse_args()

model = YOLO(args.model).to("cuda")
cap = cv2.VideoCapture(args.video)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
if abs(args.rotate) == 90:
    w, h = h, w

out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

centers, all_cycles, current_cycle, trail_points, bbox_history = [], [], [], [], []
is_going_down, min_cycle_length, fade_duration, bbox_smooth = None, 30, 60, 5


def smooth(arr):
    if len(arr) >= 7:
        win = min(11, len(arr) - (1 - len(arr) % 2))
        return savgol_filter(arr, win, 3, mode="interp")
    return arr


while True:
    ret, frame = cap.read()
    if not ret:
        break

    if args.rotate == 90:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif args.rotate == -90:
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    results = model(frame, conf=0.25, iou=0.45, verbose=False)[0]
    boxes = results.boxes.xyxy.cpu().numpy()
    frame_num = len(centers)

    if len(boxes) > 0:
        bbox_history.append(boxes[np.argmax(results.boxes.conf.cpu().numpy())])
        if len(bbox_history) > bbox_smooth:
            bbox_history.pop(0)
        x1, y1, x2, y2 = (
            np.array(bbox_history).mean(axis=0)
            if len(bbox_history) >= 3
            else bbox_history[-1]
        )

        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        centers.append((cx, cy))
        current_cycle.append((cx, cy))
        trail_points.append({"p": (cx, cy), "f": frame_num})

        if len(centers) >= 20:
            trend = centers[-1][1] - centers[-10][1]
            if trend > 10 and is_going_down != True:
                if is_going_down == False and len(current_cycle) > min_cycle_length:
                    all_cycles.append(current_cycle.copy())
                    current_cycle = [(cx, cy)]
                is_going_down = True
            elif trend < -10 and is_going_down != False:
                is_going_down = False

        cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)

    trail_points = [p for p in trail_points if frame_num - p["f"] < fade_duration]
    if len(trail_points) > 5:
        xs = smooth(np.array([p["p"][0] for p in trail_points]))
        ys = smooth(np.array([p["p"][1] for p in trail_points]))
        ages = np.array([frame_num - p["f"] for p in trail_points])

        for i in range(len(xs) - 1):
            alpha = 1.0 - ages[i] / fade_duration
            color = (
                (0, int(255 * alpha), 0)
                if ys[i + 1] > ys[i]
                else (0, 0, int(255 * alpha))
            )
            cv2.line(
                frame,
                (int(xs[i]), int(ys[i])),
                (int(xs[i + 1]), int(ys[i + 1])),
                color,
                max(2, int(2 * alpha)),
                cv2.LINE_AA,
            )

    cv2.putText(
        frame,
        f"Cykl: {len(all_cycles) + 1}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )
    out.write(frame)

if len(current_cycle) > min_cycle_length:
    all_cycles.append(current_cycle)

if len(all_cycles) > 0:
    print(f"Wykryto {len(all_cycles)} cykli.")

    down_phases, up_phases = [], []
    for cycle in all_cycles:
        if len(cycle) < 10:
            continue
        lowest = np.argmax([p[1] for p in cycle])
        if 5 < lowest < len(cycle) - 5:
            down_phases.append(cycle[: lowest + 1])
            up_phases.append(cycle[lowest:])

    def avg_phase(phases):
        if not phases:
            return None
        max_len = max(len(p) for p in phases)
        norm = []
        for p in phases:
            xs = np.interp(
                np.linspace(0, len(p) - 1, max_len), range(len(p)), [pt[0] for pt in p]
            )
            ys = np.interp(
                np.linspace(0, len(p) - 1, max_len), range(len(p)), [pt[1] for pt in p]
            )
            norm.append((xs, ys))
        avg_x = np.mean([n[0] for n in norm], axis=0)
        avg_y = np.mean([n[1] for n in norm], axis=0)
        return smooth(avg_x), smooth(avg_y)

    avg_down, avg_up = avg_phase(down_phases), avg_phase(up_phases)

    cap2 = cv2.VideoCapture(args.video)
    cap2.set(cv2.CAP_PROP_POS_FRAMES, cap2.get(cv2.CAP_PROP_FRAME_COUNT) - 1)
    ret, last_frame = cap2.read()
    cap2.release()

    if ret and avg_down is not None and avg_up is not None:
        if args.rotate in [90, -90]:
            last_frame = cv2.rotate(
                last_frame,
                cv2.ROTATE_90_CLOCKWISE
                if args.rotate == 90
                else cv2.ROTATE_90_COUNTERCLOCKWISE,
            )

        summary = cv2.addWeighted(last_frame, 0.3, np.zeros_like(last_frame), 0.7, 0)

        for (xs, ys), color in [(avg_down, (0, 255, 0)), (avg_up, (0, 0, 255))]:
            for i in range(len(xs) - 1):
                cv2.line(
                    summary,
                    (int(xs[i]), int(ys[i])),
                    (int(xs[i + 1]), int(ys[i + 1])),
                    color,
                    3,
                    cv2.LINE_AA,
                )

        cv2.putText(
            summary,
            f"USREDNIONA SCIEZKA ({len(all_cycles)} cykli)",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

        for _ in range(int(fps * 10)):
            out.write(summary)

cap.release()
out.release()
print("Zapisano:", args.output)
