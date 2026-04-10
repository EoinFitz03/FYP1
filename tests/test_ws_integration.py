"""
Integration test for the FastAPI WebSocket endpoint.

This checks that the /ws route accepts frame messages, decodes them,
passes them through the per-connection SessionState, and returns the
expected JSON payload over multiple frames.
"""

import os
import sys
import json

import numpy as np
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "backend"))

import backend.app as app_module


class FakeFaceService:
    def recognize_person(self, bgr):
        return {"person": "Owin", "face_conf": 0.97, "distance": 0.12}


class FakeGestureService:
    def predict_trained_gesture(self, bgr):
        return {"gesture": "Wave", "gesture_conf": 0.95}

    def detect_gesture_fast(self, bgr):
        return {"gesture": "—", "gesture_conf": 0.0}


class FakeEnrolService:
    active = False


def test_ws_frame_flow_returns_smoothed_face_and_gesture(monkeypatch):
    """
    Send repeated frame messages through the real /ws endpoint.

    Expectations:
    - face appears after the configured face cadence (every 2nd frame)
    - gesture appears after enough votes are accumulated (every 3rd frame,
      minimum 2 votes => visible by frame 6)
    - returned payload contains the expected integration output structure
    """
    dummy_bgr = np.zeros((480, 640, 3), dtype=np.uint8)

    # Avoid startup/shutdown side effects; we inject fake services directly.
    app_module.app.router.on_startup.clear()
    app_module.app.router.on_shutdown.clear()

    app_module.face_svc = FakeFaceService()
    app_module.gesture_svc = FakeGestureService()
    app_module.enrol_svc = FakeEnrolService()

    monkeypatch.setattr(app_module, "decode_base64_jpeg", lambda _: dummy_bgr)
    monkeypatch.setattr(app_module, "try_capture_landmarks", lambda **kwargs: (False, False, False))

    with TestClient(app_module.app) as client:
        with client.websocket_connect("/ws") as ws:
            results = []

            for _ in range(6):
                ws.send_text(json.dumps({"type": "frame", "data": "fake_base64"}))
                msg = json.loads(ws.receive_text())
                results.append(msg)

    assert all(msg["type"] == "result" for msg in results)

    first_payload = results[0]["payload"]
    final_payload = results[-1]["payload"]

    # Frame 1: nothing has reached the configured cadence yet.
    assert first_payload["person"] == "Unknown"
    assert first_payload["gesture"] == "—"

    # By frame 6: face cadence + gesture vote smoothing should both be satisfied.
    assert final_payload["person"] == "Owin"
    assert final_payload["face_conf"] == 0.97
    assert final_payload["distance"] == 0.12
    assert final_payload["gesture"] == "Wave"
    assert final_payload["gesture_conf"] == 1.0
    assert "latency_ms" in final_payload
    assert "ts" in final_payload