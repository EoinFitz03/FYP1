import os
import sys
import json
import time
import base64
from typing import Any, Dict, List, Tuple, Optional

import cv2
import numpy as np
import face_recognition

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# ----------------------------
# Paths: reuse Demo1 code + DB
# ----------------------------
APP_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(APP_DIR, ".."))
DEMO1_DIR = os.path.join(ROOT_DIR, "Demo1")

# Allow importing Demo1/db.py
sys.path.insert(0, DEMO1_DIR)
from db import Database  # Demo1/db.py

DB_PATH = os.path.join(DEMO1_DIR, "system.db")

# ----------------------------
# Tuning parameters
# ----------------------------
TOLERANCE = 0.50     # lower = stricter match; 0.45–0.60 typical
DOWNSCALE = 0.50     # speed boost for CPU; try 0.25 if slow
MODEL = "hog"        # "hog" is CPU-friendly; "cnn" needs more setup

app = FastAPI()

# React dev server origin (Vite)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

known_encodings: List[np.ndarray] = []
known_names: List[str] = []


def load_known_faces_from_db(db_path: str) -> Tuple[List[np.ndarray], List[str]]:
    db = Database(db_path)
    encs, names = db.load_all_encodings()
    # Ensure numpy arrays
    encs = [np.asarray(e) for e in encs]
    names = [str(n) for n in names]
    return encs, names


@app.on_event("startup")
def startup() -> None:
    global known_encodings, known_names
    if not os.path.exists(DB_PATH):
        print(f"[backend] ERROR: DB not found at {DB_PATH}")
        known_encodings, known_names = [], []
        return

    try:
        known_encodings, known_names = load_known_faces_from_db(DB_PATH)
        print(f"[backend] Loaded {len(known_encodings)} encodings from {DB_PATH}")
        if len(known_encodings) == 0:
            print("[backend] No encodings in DB yet. Run Demo1 enrol flow first.")
    except Exception as e:
        print(f"[backend] Failed loading encodings: {e}")
        known_encodings, known_names = [], []


def decode_base64_jpeg(base64_jpeg: str) -> Optional[np.ndarray]:
    """Return BGR image (OpenCV) from base64 JPEG."""
    try:
        jpg_bytes = base64.b64decode(base64_jpeg)
        arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return bgr
    except Exception:
        return None


def recognize_person(bgr: np.ndarray) -> Dict[str, Any]:
    """Run face recognition and return best match."""
    if len(known_encodings) == 0:
        return {"person": "Unknown", "face_conf": 0.0, "distance": None, "note": "No known encodings loaded"}

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Downscale for speed
    if DOWNSCALE != 1.0:
        rgb_small = cv2.resize(rgb, (0, 0), fx=DOWNSCALE, fy=DOWNSCALE)
    else:
        rgb_small = rgb

    locations = face_recognition.face_locations(rgb_small, model=MODEL)
    if not locations:
        return {"person": "Unknown", "face_conf": 0.0, "distance": None}

    encs = face_recognition.face_encodings(rgb_small, locations)
    if not encs:
        return {"person": "Unknown", "face_conf": 0.0, "distance": None}

    # Choose the best match across all faces found
    best_name = "Unknown"
    best_dist = 999.0

    for enc in encs:
        distances = face_recognition.face_distance(known_encodings, enc)
        if len(distances) == 0:
            continue
        i = int(np.argmin(distances))
        d = float(distances[i])
        if d < best_dist:
            best_dist = d
            best_name = known_names[i] if d <= TOLERANCE else "Unknown"

    if best_dist == 999.0:
        return {"person": "Unknown", "face_conf": 0.0, "distance": None}

    # Rough confidence mapping: 1.0 at dist=0, ~0 at dist=tolerance
    conf = max(0.0, min(1.0, 1.0 - (best_dist / TOLERANCE)))
    return {"person": best_name, "face_conf": round(conf, 3), "distance": round(best_dist, 4)}


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            raw = await ws.receive_text()
            t0 = time.perf_counter()

            # Parse incoming message
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                msg = {"type": "ping"}

            if msg.get("type") != "frame":
                # Keepalive response (optional)
                await ws.send_text(json.dumps({
                    "type": "result",
                    "payload": {"person": "—", "gesture": "—", "face_conf": 0.0, "gesture_conf": 0.0, "latency_ms": 0}
                }))
                continue

            base64jpeg = msg.get("data", "")
            bgr = decode_base64_jpeg(base64jpeg)
            if bgr is None:
                await ws.send_text(json.dumps({
                    "type": "result",
                    "payload": {"person": "Unknown", "gesture": "—", "face_conf": 0.0, "gesture_conf": 0.0, "latency_ms": 0, "error": "decode_failed"}
                }))
                continue

            face_res = recognize_person(bgr)

            latency_ms = (time.perf_counter() - t0) * 1000.0
            payload = {
                "person": face_res["person"],
                "face_conf": face_res["face_conf"],
                "gesture": "—",          # Week 3: fill from MediaPipe
                "gesture_conf": 0.0,
                "latency_ms": round(latency_ms, 1),
                "distance": face_res.get("distance"),
                "ts": time.time(),
            }

            await ws.send_text(json.dumps({"type": "result", "payload": payload}))

    except WebSocketDisconnect:
        return
