import csv
import os
from typing import Iterable

# backend/...
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
DATASET_PATH = os.path.join(DATASET_DIR, "gestures.csv")


def ensure_csv_exists(path: str = DATASET_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if os.path.exists(path):
        return

    header = ["label", "hand"]
    for i in range(21):
        header += [f"x{i}", f"y{i}", f"z{i}"]

    with open(path, "w", newline="") as f:
        csv.writer(f).writerow(header)


def append_sample(label: str, hand: str, landmarks: Iterable, path: str = DATASET_PATH) -> None:
    """
    landmarks: MediaPipe landmarks list (21 items). Each item has .x .y .z
    """
    ensure_csv_exists(path)

    row = [label, hand]
    for lm in landmarks:
        row += [float(lm.x), float(lm.y), float(lm.z)]

    with open(path, "a", newline="") as f:
        csv.writer(f).writerow(row)