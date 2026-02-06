import os
import sys
import json
import time
from typing import Any, Dict, List
from collections import deque, Counter

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# ----------------------------
# Paths: reuse Demo1 code + DB
# ----------------------------
APP_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(APP_DIR, ".."))
DEMO1_DIR = os.path.join(ROOT_DIR, "Demo1")
HAND_DIR = os.path.join(ROOT_DIR, "HandGestures")

# Make Demo1 + HandGestures importable 
sys.path.insert(0, DEMO1_DIR)
sys.path.insert(0, HAND_DIR)

DB_PATH = os.path.join(DEMO1_DIR, "system.db")

# ----------------------------
# Real-time tuning parameters 
# ----------------------------
TOLERANCE = 0.50
DOWNSCALE = 0.50
MODEL = "hog"

FACE_EVERY_N_FRAMES = 2
GESTURE_SMALL_WIDTH = 320
GESTURE_SMOOTH_WINDOW = 5
GESTURE_MIN_VOTES = 2

HAND_LOST_MS = 1200
HAND_MISS_CLEAR_COUNT = 6
GESTURE_EVERY_N_FRAMES = 3
FACE_LOST_MS = 800

# Enrol capture spacing 
ENROL_MIN_MS_BETWEEN_CAPTURES = 250

# ----------------------------
# Services
# ----------------------------
from services.frame_service import decode_base64_jpeg
from services.face_service import FaceService
from services.gesture_service import GestureService
from services.enrol_service import EnrolService


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global services loaded once, reused by websocket
face_svc: FaceService | None = None
gesture_svc: GestureService | None = None
enrol_svc: EnrolService | None = None


@app.on_event("startup")
def startup() -> None:
    global face_svc, gesture_svc, enrol_svc

    face_svc = FaceService(
        db_path=DB_PATH,
        tolerance=TOLERANCE,
        downscale=DOWNSCALE,
        model=MODEL,
    )
    face_svc.load_known_faces()  # loads encodings into memory 

    gesture_svc = GestureService(
        gesture_small_width=GESTURE_SMALL_WIDTH
    )
    gesture_svc.startup()  # sets up mediapipe hands if available

    enrol_svc = EnrolService(
        db_path=DB_PATH,
        model=MODEL,
        min_ms_between_captures=ENROL_MIN_MS_BETWEEN_CAPTURES,
        on_saved=face_svc.load_known_faces,  # reload encodings after enroll
    )


@app.on_event("shutdown")
def shutdown() -> None:
    global gesture_svc
    if gesture_svc is not None:
        gesture_svc.shutdown()


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()

    assert face_svc is not None, "FaceService not initialised"
    assert gesture_svc is not None, "GestureService not initialised"
    assert enrol_svc is not None, "EnrolService not initialised"

    frame_i = 0

    # Reuse last results between heavy runs
    last_face: Dict[str, Any] = {"person": "Unknown", "face_conf": 0.0, "distance": None}
    last_face_seen = 0.0

    last_gesture: Dict[str, Any] = {"gesture": "—", "gesture_conf": 0.0}
    gesture_hist = deque(maxlen=GESTURE_SMOOTH_WINDOW)

    last_hand_seen = 0.0
    hand_miss_count = 0

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
            # Enrol controls 
            # ----------------------------
            if mtype == "enrol_start":
                name = str(msg.get("name", "")).strip()
                target = int(msg.get("num_samples", 10))
                status = enrol_svc.start(name=name, num_samples=target)

                await ws.send_text(json.dumps({"type": "enrol_status", "payload": status}))
                continue

            if mtype == "enrol_cancel":
                status = enrol_svc.cancel()
                await ws.send_text(json.dumps({"type": "enrol_status", "payload": status}))
                continue

            # If not a frame, return current snapshot (UNCHANGED)
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
            # Enrol capture 
            # ----------------------------
            if enrol_svc.active:
                enrol_update = enrol_svc.try_capture(bgr)
                if enrol_update is not None:
                    await ws.send_text(json.dumps({"type": "enrol_status", "payload": enrol_update}))

            # -------- Face every N frames --------
            if frame_i % FACE_EVERY_N_FRAMES == 0:
                new_face = face_svc.recognize_person(bgr)

                if new_face["person"] != "Unknown":
                    last_face = new_face
                    last_face_seen = time.time()
                else:
                    if last_face["person"] != "Unknown" and (time.time() - last_face_seen) * 1000.0 <= FACE_LOST_MS:
                        pass
                    else:
                        last_face = new_face

            # -------- Gesture every N frame --------
            if frame_i % GESTURE_EVERY_N_FRAMES == 0:
                raw_g = gesture_svc.detect_gesture_fast(bgr)

                if raw_g["gesture"] == "—":
                    hand_miss_count += 1
                    if hand_miss_count >= HAND_MISS_CLEAR_COUNT:
                        last_gesture = {"gesture": "—", "gesture_conf": 0.0}
                        gesture_hist.clear()
                else:
                    hand_miss_count = 0
                    last_hand_seen = time.time()
                    gesture_hist.append(raw_g["gesture"])

                    counts = Counter(gesture_hist)
                    best_gesture, best_votes = counts.most_common(1)[0]

                    if best_gesture != "—" and best_votes >= GESTURE_MIN_VOTES:
                        last_gesture = {
                            "gesture": best_gesture,
                            "gesture_conf": round(best_votes / len(gesture_hist), 3),
                        }

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
