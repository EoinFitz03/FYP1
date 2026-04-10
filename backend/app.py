import os 
import sys
import json # used to read messages from the fontend 
import time # used for time latency 

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect # hnadles front end connection 
from fastapi.middleware.cors import CORSMiddleware

# Paths

# Build important folder paths and make Demo1 / HandGestures importable
APP_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(APP_DIR, ".."))
DEMO1_DIR = os.path.join(ROOT_DIR, "Demo1")
HAND_DIR = os.path.join(ROOT_DIR, "HandGestures")

sys.path.insert(0, DEMO1_DIR)
sys.path.insert(0, HAND_DIR)

DB_PATH = os.path.join(DEMO1_DIR, "system.db") # path to SQL lite database 


# App + services

from config import cfg 
from session import SessionState # stores live session info and controls per-frame logic
from services.frame_service import decode_base64_jpeg # 
from services.face_service import FaceService
from services.gesture_service import GestureService
from services.enrol_service import EnrolService # Import configuration and backend services used during live processing

# Step 1 training capture
from training.capture import TrainingState, try_capture_landmarks # stores training-capture progress
# saves gesture landmark data during training capture
app = FastAPI()
app.add_middleware(
    CORSMiddleware, # is needed because frontend and backend run on different ports
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
) # Create FastAPI app and allow requests from the local React frontend

# Section 5 
face_svc: FaceService | None = None
gesture_svc: GestureService | None = None
enrol_svc: EnrolService | None = None # Global service instances created when the backend starts

# Step 1 training capture
train_state = TrainingState() # Stores the current training capture session state

# (6)
@app.on_event("startup") 
# creates
# face recognition service
# gesture recognition service
# enrolment service 
def startup() -> None:
    global face_svc, gesture_svc, enrol_svc

    face_svc = FaceService( # Create face recognition service and load known face encodings from the database
        db_path=DB_PATH,
        tolerance=cfg.tolerance,
        downscale=cfg.downscale,
        model=cfg.model,
    )
    face_svc.load_known_faces()

    gesture_svc = GestureService(gesture_small_width=cfg.gesture_small_width) # Create gesture recognition service and initialise MediaPipe / trained model resources
    gesture_svc.startup()

    enrol_svc = EnrolService( # Create enrolment service for capturing and saving new user face samples
        db_path=DB_PATH,
        model=cfg.model,
        min_ms_between_captures=cfg.enrol_min_ms_between_captures,
        on_saved=face_svc.load_known_faces,
    )

# (7)
@app.on_event("shutdown")
def shutdown() -> None:
    if gesture_svc is not None:
        gesture_svc.shutdown()
# Clean up gesture resources when the backend shuts down
#(8)
@app.websocket("/ws") # Main WebSocket endpoint used for live communication with the frontend
async def ws_endpoint(ws: WebSocket):
    await ws.accept() # Accept the incoming frontend WebSocket connection

    # Make sure required services exist and create a session for this connected client
    assert face_svc and gesture_svc and enrol_svc, "Services not initialised" 

    session = SessionState()
# (9)
    try:
        while True:# Continuously receive messages from the frontend while the WebSocket is connected
            raw = await ws.receive_text() # receive_text() gets the next message from frontend
            t0 = time.perf_counter() # stores the start time so latency can be calculated later
# (10)
            try:             # Parse incoming JSON message and read its type
                msg = json.loads(raw)
            except json.JSONDecodeError:
                msg = {"type": "ping"}  # If parsing fails, treat it like a simple ping message

            mtype = msg.get("type")

            # --- Enrol controls --- (12)
            if mtype == "enrol_start":  # Start face enrolment when requested by the frontend
                name = str(msg.get("name", "")).strip() # get sthe name from the forntend 
                target = int(msg.get("num_samples", 10)) # get steh number of samples
                status = enrol_svc.start(name=name, num_samples=target)
                await ws.send_text(json.dumps({"type": "enrol_status", "payload": status}))
                continue

            if mtype == "enrol_cancel": # Cancel the current enrolment session
                status = enrol_svc.cancel()
                await ws.send_text(json.dumps({"type": "enrol_status", "payload": status}))
                continue

            # --- Training controls Step 1 dataset capture---
            if mtype == "train_start": # Start gesture training capture and reset the training state
                label = str(msg.get("label", "")).strip()
                target = int(msg.get("num_samples", 200))

                if not label: # Reject training start if no gesture label was provided
                    await ws.send_text(json.dumps({
                        "type": "train_status",
                        "payload": {"active": False, "error": "missing_label"},
                    }))
                    continue
                # Store the new training session details
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
            # Stop the current gesture training capture session
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

            # --- Non-frame messages: return current snapshot --- (14)
            if mtype != "frame": # For non-frame messages, return the current session snapshot instead of running recognition
                payload = session.build_snapshot()
                await ws.send_text(json.dumps({"type": "result", "payload": payload}))
                continue

            # --- Decode frame --- (15)
            bgr = decode_base64_jpeg(msg.get("data", ""))# Decode the base64 JPEG frame sent by the frontend into an OpenCV BGR image
            if bgr is None: # If decoding fails, send back the current snapshot with an error flag
                await ws.send_text(json.dumps({
                    "type": "result",
                    "payload": {**session.build_snapshot(), "error": "decode_failed"},
                }))
                continue

            # --- Enrol capture --- (16)
            if enrol_svc.active: # If enrolment is active, try to capture a face sample from this frame
                enrol_update = enrol_svc.try_capture(bgr)
                if enrol_update is not None:
                    await ws.send_text(json.dumps({"type": "enrol_status", "payload": enrol_update}))

            # --- Training capture (Step 1): save landmarks to CSV --- (17)
            did_save, should_send, finished = try_capture_landmarks( # Try to capture gesture landmarks from the current frame for training data collection
                bgr=bgr,
                cfg=cfg,
                gesture_svc=gesture_svc,
                state=train_state,
            )
            if should_send: # Send updated training progress back to the frontend when needed
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

            # --- Process frame (face + gesture) ---  # Run the main per-frame processing pipeline for face and gesture recognition
            payload = session.process_frame(bgr, face_svc, gesture_svc)
            payload["latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
            # Send the processed frame result back to the frontend
            await ws.send_text(json.dumps({"type": "result", "payload": payload}))

    except WebSocketDisconnect:     # Exit cleanly when the frontend WebSocket disconnects
        return