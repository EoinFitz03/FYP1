import os
import sys
import json
import time
import base64
from typing import Any, Dict, List, Tuple, Optional
from collections import deque, Counter

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

sys.path.insert(0, DEMO1_DIR)
from db import Database  # Demo1/db.py

DB_PATH = os.path.join(DEMO1_DIR, "system.db")

# ----------------------------
# Hand Gestures (reuse your existing HandGestures demo)
# ----------------------------
HAND_DIR = os.path.join(ROOT_DIR, "HandGestures")
sys.path.insert(0, HAND_DIR)

try:
    from gestures_live import (
        mp_hands,
        classify_gesture,
        Gesture,
        MIN_DET_CONF,
        MIN_TRK_CONF,
        MODEL_COMPLEXITY,
        MAX_HANDS,
    )
except Exception as e:
    mp_hands = None
    classify_gesture = None
    Gesture = None
    MIN_DET_CONF = 0.5
    MIN_TRK_CONF = 0.5
    MODEL_COMPLEXITY = 0
    MAX_HANDS = 1
    print(f"[backend] Gesture imports failed: {e}")

# ----------------------------
# Tuning parameters
# ----------------------------
TOLERANCE = 0.50
DOWNSCALE = 0.50
MODEL = "hog"

# ----------------------------
# Real-time performance controls
# ----------------------------
FACE_EVERY_N_FRAMES = 2          # run face recognition every 2 frames, reuse last in-between
GESTURE_EVERY_N_FRAMES = 4       # run gesture detection every 4 frames, reuse last in-between
GESTURE_SMALL_WIDTH = 320        # run MediaPipe on resized image
GESTURE_SMOOTH_WINDOW = 5        # vote smoothing window
GESTURE_MIN_VOTES = 2            # must appear at least this many times in window

HAND_LOST_MS = 500               # clear gesture ~0.5s after last detected hand

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

known_encodings: List[np.ndarray] = []
known_names: List[str] = []
hands_detector = None


def load_known_faces_from_db(db_path: str) -> Tuple[List[np.ndarray], List[str]]:
    db = Database(db_path)
    encs, names = db.load_all_encodings()
    encs = [np.asarray(e) for e in encs]
    names = [str(n) for n in names]
    return encs, names


@app.on_event("startup")
def startup() -> None:
    global known_encodings, known_names, hands_detector

    # Load face encodings
    if not os.path.exists(DB_PATH):
        print(f"[backend] ERROR: DB not found at {DB_PATH}")
        known_encodings, known_names = [], []
    else:
        try:
            known_encodings, known_names = load_known_faces_from_db(DB_PATH)
            print(f"[backend] Loaded {len(known_encodings)} encodings from {DB_PATH}")
        except Exception as e:
            print(f"[backend] Failed loading encodings: {e}")
            known_encodings, known_names = [], []

    # Init MediaPipe Hands
    try:
        if mp_hands is not None:
            hands_detector = mp_hands.Hands(
                model_complexity=0,          # IMPORTANT: 0 is fastest
                max_num_hands=1,             # IMPORTANT: one hand for speed
                min_detection_confidence=MIN_DET_CONF,
                min_tracking_confidence=MIN_TRK_CONF,
            )
            print("[backend] MediaPipe Hands initialised")
        else:
            hands_detector = None
            print("[backend] MediaPipe Hands not available (imports failed)")
    except Exception as e:
        hands_detector = None
        print(f"[backend] Failed to initialise MediaPipe Hands: {e}")


@app.on_event("shutdown")
def shutdown() -> None:
    global hands_detector
    try:
        if hands_detector is not None:
            hands_detector.close()
            hands_detector = None
    except Exception:
        pass


def decode_base64_jpeg(base64_jpeg: str) -> Optional[np.ndarray]:
    try:
        jpg_bytes = base64.b64decode(base64_jpeg)
        arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


def recognize_person(bgr: np.ndarray) -> Dict[str, Any]:
    if len(known_encodings) == 0:
        return {"person": "Unknown", "face_conf": 0.0, "distance": None}

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

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

    conf = max(0.0, min(1.0, 1.0 - (best_dist / TOLERANCE)))
    return {"person": best_name, "face_conf": round(conf, 3), "distance": round(best_dist, 4)}


def detect_gesture_fast(bgr: np.ndarray) -> Dict[str, Any]:
    """Run MediaPipe on a smaller image for speed, then classify."""
    if hands_detector is None or classify_gesture is None:
        return {"gesture": "—", "gesture_conf": 0.0}

    h, w = bgr.shape[:2]
    if w > GESTURE_SMALL_WIDTH:
        scale = GESTURE_SMALL_WIDTH / float(w)
        small = cv2.resize(bgr, (GESTURE_SMALL_WIDTH, int(h * scale)))
    else:
        small = bgr

    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    res = hands_detector.process(rgb)

    if not res.multi_hand_landmarks:
        return {"gesture": "—", "gesture_conf": 0.0}

    try:
        g = classify_gesture(res.multi_hand_landmarks[0])
        if Gesture is not None and g == Gesture.UNKNOWN:
            return {"gesture": "—", "gesture_conf": 0.0}

        label = str(g.value) if hasattr(g, "value") else str(g)
        return {"gesture": label, "gesture_conf": 1.0}
    except Exception:
        return {"gesture": "—", "gesture_conf": 0.0}


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()

    frame_i = 0

    # Reuse last results between heavy runs
    last_face = {"person": "Unknown", "face_conf": 0.0, "distance": None}
    last_gesture = {"gesture": "—", "gesture_conf": 0.0}

    # Smooth gestures
    gesture_hist = deque(maxlen=GESTURE_SMOOTH_WINDOW)

    # For clearing “stuck” gestures
    last_hand_seen = 0.0

    try:
        while True:
            raw = await ws.receive_text()
            t0 = time.perf_counter()

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                msg = {"type": "ping"}

            if msg.get("type") != "frame":
                await ws.send_text(json.dumps({
                    "type": "result",
                    "payload": {
                        "person": last_face["person"],
                        "face_conf": last_face["face_conf"],
                        "distance": last_face.get("distance"),
                        "gesture": last_gesture["gesture"],
                        "gesture_conf": last_gesture["gesture_conf"],
                        "latency_ms": 0,
                        "ts": time.time(),
                    }
                }))
                continue

            frame_i += 1

            base64jpeg = msg.get("data", "")
            bgr = decode_base64_jpeg(base64jpeg)
            if bgr is None:
                await ws.send_text(json.dumps({
                    "type": "result",
                    "payload": {
                        "person": "Unknown",
                        "face_conf": 0.0,
                        "distance": None,
                        "gesture": "—",
                        "gesture_conf": 0.0,
                        "latency_ms": 0,
                        "error": "decode_failed",
                        "ts": time.time(),
                    }
                }))
                continue

            # -------- Face (every N frames) --------
            if frame_i % FACE_EVERY_N_FRAMES == 0:
                last_face = recognize_person(bgr)
            # else reuse last_face

            # -------- Gesture (every N frames) --------
            if frame_i % GESTURE_EVERY_N_FRAMES == 0:
                raw_g = detect_gesture_fast(bgr)

                # If no hand detected, clear quickly (prevents “sticking”)
                if raw_g["gesture"] == "—":
                    gesture_hist.clear()
                else:
                    last_hand_seen = time.time()
                    gesture_hist.append(raw_g["gesture"])

                # vote smoothing
                if len(gesture_hist) == 0:
                    last_gesture = {"gesture": "—", "gesture_conf": 0.0}
                else:
                    counts = Counter(gesture_hist)
                    best_gesture, best_votes = counts.most_common(1)[0]

                    if best_gesture == "—" or best_votes < GESTURE_MIN_VOTES:
                        last_gesture = {"gesture": "—", "gesture_conf": 0.0}
                    else:
                        last_gesture = {
                            "gesture": best_gesture,
                            "gesture_conf": round(best_votes / len(gesture_hist), 3),
                        }

            # Extra safety: if we haven't seen a hand recently, force clear
            if last_gesture["gesture"] != "—":
                if last_hand_seen == 0.0 or (time.time() - last_hand_seen) * 1000.0 > HAND_LOST_MS:
                    last_gesture = {"gesture": "—", "gesture_conf": 0.0}
                    gesture_hist.clear()

            latency_ms = (time.perf_counter() - t0) * 1000.0

            payload = {
                "person": last_face["person"],
                "face_conf": last_face["face_conf"],
                "distance": last_face.get("distance"),
                "gesture": last_gesture["gesture"],
                "gesture_conf": last_gesture["gesture_conf"],
                "latency_ms": round(latency_ms, 1),
                "ts": time.time(),
            }

            await ws.send_text(json.dumps({"type": "result", "payload": payload}))

    except WebSocketDisconnect:
        return
