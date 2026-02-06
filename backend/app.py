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
# Face tuning parameters
# ----------------------------
TOLERANCE = 0.50
DOWNSCALE = 0.50
MODEL = "hog"

# ----------------------------
# Real-time performance controls
# ----------------------------
FACE_EVERY_N_FRAMES = 2          # run face recognition every 2 frames
GESTURE_EVERY_N_FRAMES = 4       # run gesture detection every 4 frames
GESTURE_SMALL_WIDTH = 320        # run MediaPipe on resized image
GESTURE_SMOOTH_WINDOW = 5        # vote smoothing window
GESTURE_MIN_VOTES = 2            # must appear at least this many times in window

# "Hold" logic to prevent flicker
HAND_LOST_MS = 1200           # hold gesture longer before clearing
HAND_MISS_CLEAR_COUNT = 6     # need more consecutive misses to clear
GESTURE_EVERY_N_FRAMES = 3    # check gestures a bit more often
FACE_LOST_MS = 800  # keep last face ID for 0.8s when face is briefly lost (hand up / occlusion)


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


def extract_single_face_encoding(bgr: np.ndarray) -> Optional[np.ndarray]:
    """
    Enrol helper: take one incoming frame and return ONE face encoding.
    Reject frames with 0 faces or >1 face to avoid enrolling the wrong person.
    """
    small = cv2.resize(bgr, (0, 0), fx=0.25, fy=0.25)
    rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    locs = face_recognition.face_locations(rgb_small, model=MODEL)
    if not locs:
        return None

    encs = face_recognition.face_encodings(rgb_small, locs)
    if not encs:
        return None

    if len(encs) != 1:
        return None

    return encs[0]


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()

    frame_i = 0

    # Reuse last results between heavy runs
    last_face = {"person": "Unknown", "face_conf": 0.0, "distance": None}
    last_face_seen = 0.0

    last_gesture = {"gesture": "—", "gesture_conf": 0.0}
    gesture_hist = deque(maxlen=GESTURE_SMOOTH_WINDOW)

    last_hand_seen = 0.0
    hand_miss_count = 0

    # ----------------------------
    # Enrol state 
    # ----------------------------
    enrol_active = False
    enrol_name = ""
    enrol_target = 10
    enrol_collected: List[np.ndarray] = []
    enrol_last_capture = 0.0
    ENROL_MIN_MS_BETWEEN_CAPTURES = 250

    try:
        while True:
            raw = await ws.receive_text()
            t0 = time.perf_counter()

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                msg = {"type": "ping"}

            mtype = msg.get("type")

            # ----------------------------
            # Enrol controls (new)
            # ----------------------------
            if mtype == "enrol_start":
                enrol_name = str(msg.get("name", "")).strip()
                enrol_target = int(msg.get("num_samples", 10))
                enrol_target = max(3, min(30, enrol_target))

                enrol_collected = []
                enrol_active = bool(enrol_name)
                enrol_last_capture = 0.0

                await ws.send_text(json.dumps({
                    "type": "enrol_status",
                    "payload": {
                        "active": enrol_active,
                        "name": enrol_name,
                        "captured": 0,
                        "target": enrol_target,
                        "done": False,
                        "error": None if enrol_active else "missing_name",
                        "ts": time.time(),
                    }
                }))
                continue

            if mtype == "enrol_cancel":
                enrol_active = False
                enrol_name = ""
                enrol_collected = []
                enrol_last_capture = 0.0

                await ws.send_text(json.dumps({
                    "type": "enrol_status",
                    "payload": {
                        "active": False,
                        "name": "",
                        "captured": 0,
                        "target": 0,
                        "done": False,
                        "error": "cancelled",
                        "ts": time.time(),
                    }
                }))
                continue

            # Existing behaviour: if not a frame, return current result snapshot
            if mtype != "frame":
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

            # ----------------------------
            # Enrol capture (new)
            # ----------------------------
            if enrol_active:
                now = time.time()
                if (now - enrol_last_capture) * 1000.0 >= ENROL_MIN_MS_BETWEEN_CAPTURES:
                    enc = extract_single_face_encoding(bgr)
                    if enc is not None:
                        enrol_collected.append(enc)
                        enrol_last_capture = now

                        await ws.send_text(json.dumps({
                            "type": "enrol_status",
                            "payload": {
                                "active": True,
                                "name": enrol_name,
                                "captured": len(enrol_collected),
                                "target": enrol_target,
                                "done": False,
                                "error": None,
                                "ts": time.time(),
                            }
                        }))

                        if len(enrol_collected) >= enrol_target:
                            mean_encoding = np.mean(enrol_collected, axis=0)

                            db = Database(DB_PATH)
                            user_id = db.get_user_id(enrol_name)
                            if user_id is None:
                                user_id = db.add_user(enrol_name)
                            db.add_face_encoding(user_id, mean_encoding)
                            db.close()

                            # reload so Live works instantly
                            global known_encodings, known_names
                            known_encodings, known_names = load_known_faces_from_db(DB_PATH)

                            enrol_active = False
                            enrol_name = ""
                            enrol_collected = []
                            enrol_last_capture = 0.0

                            await ws.send_text(json.dumps({
                                "type": "enrol_status",
                                "payload": {
                                    "active": False,
                                    "name": "",
                                    "captured": enrol_target,
                                    "target": enrol_target,
                                    "done": True,
                                    "error": None,
                                    "ts": time.time(),
                                }
                            }))

            # -------- Face (every N frames) --------
            if frame_i % FACE_EVERY_N_FRAMES == 0:
                new_face = recognize_person(bgr)

                # If we got a real person, update and mark seen time
                if new_face["person"] != "Unknown":
                    last_face = new_face
                    last_face_seen = time.time()
                else:
                    # Hold previous face briefly if it was seen recently (prevents flicker)
                    if last_face["person"] != "Unknown" and (time.time() - last_face_seen) * 1000.0 <= FACE_LOST_MS:
                        pass  # keep last_face
                    else:
                        last_face = new_face

            # -------- Gesture (every N frames) --------
            if frame_i % GESTURE_EVERY_N_FRAMES == 0:
                raw_g = detect_gesture_fast(bgr)

                if raw_g["gesture"] == "—":
                    # don't clear on one miss; require consecutive misses
                    hand_miss_count += 1
                    if hand_miss_count >= HAND_MISS_CLEAR_COUNT:
                        last_gesture = {"gesture": "—", "gesture_conf": 0.0}
                        gesture_hist.clear()
                else:
                    # saw a hand again
                    hand_miss_count = 0
                    last_hand_seen = time.time()
                    gesture_hist.append(raw_g["gesture"])

                    # vote smoothing
                    counts = Counter(gesture_hist)
                    best_gesture, best_votes = counts.most_common(1)[0]

                    if best_gesture != "—" and best_votes >= GESTURE_MIN_VOTES:
                        last_gesture = {
                            "gesture": best_gesture,
                            "gesture_conf": round(best_votes / len(gesture_hist), 3),
                        }
                    # else: keep last_gesture (don’t instantly blank)

            # Extra safety: if we haven't seen a hand recently, force clear
            if last_gesture["gesture"] != "—":
                if last_hand_seen == 0.0 or (time.time() - last_hand_seen) * 1000.0 > HAND_LOST_MS:
                    last_gesture = {"gesture": "—", "gesture_conf": 0.0}
                    gesture_hist.clear()
                    hand_miss_count = 0

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
