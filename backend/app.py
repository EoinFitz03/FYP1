import os
import sys
import json
import time

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# ----------------------------
# Paths
# ----------------------------
APP_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(APP_DIR, ".."))
DEMO1_DIR = os.path.join(ROOT_DIR, "Demo1")
HAND_DIR = os.path.join(ROOT_DIR, "HandGestures")

sys.path.insert(0, DEMO1_DIR)
sys.path.insert(0, HAND_DIR)

DB_PATH = os.path.join(DEMO1_DIR, "system.db")

# ----------------------------
# App + services
# ----------------------------
from config import cfg
from session import SessionState
from services.frame_service import decode_base64_jpeg
from services.face_service import FaceService
from services.gesture_service import GestureService
from services.enrol_service import EnrolService

# NEW (Step 1 training capture)
from training.capture import TrainingState, try_capture_landmarks

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

face_svc: FaceService | None = None
gesture_svc: GestureService | None = None
enrol_svc: EnrolService | None = None

# NEW (Step 1 training capture)
train_state = TrainingState()


@app.on_event("startup")
def startup() -> None:
    global face_svc, gesture_svc, enrol_svc

    face_svc = FaceService(
        db_path=DB_PATH,
        tolerance=cfg.tolerance,
        downscale=cfg.downscale,
        model=cfg.model,
    )
    face_svc.load_known_faces()

    gesture_svc = GestureService(gesture_small_width=cfg.gesture_small_width)
    gesture_svc.startup()

    enrol_svc = EnrolService(
        db_path=DB_PATH,
        model=cfg.model,
        min_ms_between_captures=cfg.enrol_min_ms_between_captures,
        on_saved=face_svc.load_known_faces,
    )


@app.on_event("shutdown")
def shutdown() -> None:
    if gesture_svc is not None:
        gesture_svc.shutdown()


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()

    assert face_svc and gesture_svc and enrol_svc, "Services not initialised"

    session = SessionState()

    try:
        while True:
            raw = await ws.receive_text()
            t0 = time.perf_counter()

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                msg = {"type": "ping"}

            mtype = msg.get("type")

            # --- Enrol controls ---
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

            # --- Training controls (Step 1 dataset capture) ---
            if mtype == "train_start":
                label = str(msg.get("label", "")).strip()
                target = int(msg.get("num_samples", 200))

                if not label:
                    await ws.send_text(json.dumps({
                        "type": "train_status",
                        "payload": {"active": False, "error": "missing_label"},
                    }))
                    continue

                train_state.active = True
                train_state.label = label
                train_state.target = target
                train_state.count = 0

                await ws.send_text(json.dumps({
                    "type": "train_status",
                    "payload": {
                        "active": True,
                        "label": train_state.label,
                        "count": train_state.count,
                        "target": train_state.target,
                    },
                }))
                continue

            if mtype == "train_stop":
                train_state.active = False
                train_state.label = None

                await ws.send_text(json.dumps({
                    "type": "train_status",
                    "payload": {
                        "active": False,
                        "label": None,
                        "count": train_state.count,
                        "target": train_state.target,
                    },
                }))
                continue

            # --- Non-frame messages: return current snapshot ---
            if mtype != "frame":
                payload = session.build_snapshot()
                await ws.send_text(json.dumps({"type": "result", "payload": payload}))
                continue

            # --- Decode frame ---
            bgr = decode_base64_jpeg(msg.get("data", ""))
            if bgr is None:
                await ws.send_text(json.dumps({
                    "type": "result",
                    "payload": {**session.build_snapshot(), "error": "decode_failed"},
                }))
                continue

            # --- Enrol capture ---
            if enrol_svc.active:
                enrol_update = enrol_svc.try_capture(bgr)
                if enrol_update is not None:
                    await ws.send_text(json.dumps({"type": "enrol_status", "payload": enrol_update}))

            # --- Training capture (Step 1): save landmarks to CSV ---
            did_save, should_send, finished = try_capture_landmarks(
                bgr=bgr,
                cfg=cfg,
                gesture_svc=gesture_svc,
                state=train_state,
            )
            if should_send:
                await ws.send_text(json.dumps({
                    "type": "train_status",
                    "payload": {
                        "active": train_state.active,
                        "label": train_state.label,
                        "count": train_state.count,
                        "target": train_state.target,
                        "saved": did_save,
                        "finished": finished,
                    },
                }))

            # --- Process frame (face + gesture) ---
            payload = session.process_frame(bgr, face_svc, gesture_svc)
            payload["latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)

            await ws.send_text(json.dumps({"type": "result", "payload": payload}))

    except WebSocketDisconnect:
        return