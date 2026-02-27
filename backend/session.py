"""
Per-connection session state.

Owns all the stateful logic that used to live inside ws_endpoint:
  - frame counter
  - face staleness / last-seen tracking
  - gesture smoothing / voting
  - hand-miss counter
"""
from __future__ import annotations

import time
from collections import Counter, deque
from typing import Any, Dict

import numpy as np

from config import cfg
from services.face_service import FaceService
from services.gesture_service import GestureService


_UNKNOWN_FACE: Dict[str, Any] = {"person": "Unknown", "face_conf": 0.0, "distance": None}
_EMPTY_GESTURE: Dict[str, Any] = {"gesture": "—", "gesture_conf": 0.0}


class SessionState:
    """One instance per WebSocket connection."""

    def __init__(self) -> None:
        self.frame_i: int = 0

        # Face state
        self.last_face: Dict[str, Any] = _UNKNOWN_FACE.copy()
        self.last_face_seen: float = 0.0

        # Gesture state
        self.last_gesture: Dict[str, Any] = _EMPTY_GESTURE.copy()
        self.gesture_hist: deque = deque(maxlen=cfg.gesture_smooth_window)
        self.last_hand_seen: float = 0.0
        self.hand_miss_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_frame(
        self,
        bgr: np.ndarray,
        face_svc: FaceService,
        gesture_svc: GestureService,
    ) -> Dict[str, Any]:
        """
        Run face + gesture processing on one frame.
        Returns a result payload dict ready to send over the WebSocket.
        """
        self.frame_i += 1

        if self.frame_i % cfg.face_every_n_frames == 0:
            self._update_face(bgr, face_svc)

        if self.frame_i % cfg.gesture_every_n_frames == 0:
            self._update_gesture(bgr, gesture_svc)

        self._expire_gesture()

        return self._build_payload()

    def build_snapshot(self) -> Dict[str, Any]:
        """Return current state without running any detection (for non-frame messages)."""
        return self._build_payload(latency_ms=0)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _update_face(self, bgr: np.ndarray, face_svc: FaceService) -> None:
        new_face = face_svc.recognize_person(bgr)

        if new_face["person"] != "Unknown":
            self.last_face = new_face
            self.last_face_seen = time.time()
        else:
            still_within_grace = (
                self.last_face["person"] != "Unknown"
                and (time.time() - self.last_face_seen) * 1000.0 <= cfg.face_lost_ms
            )
            if not still_within_grace:
                self.last_face = new_face

    def _update_gesture(self, bgr: np.ndarray, gesture_svc: GestureService) -> None:
        raw = gesture_svc.detect_gesture_fast(bgr)

        if raw["gesture"] == "—":
            self.hand_miss_count += 1
            if self.hand_miss_count >= cfg.hand_miss_clear_count:
                self.last_gesture = _EMPTY_GESTURE.copy()
                self.gesture_hist.clear()
        else:
            self.hand_miss_count = 0
            self.last_hand_seen = time.time()
            self.gesture_hist.append(raw["gesture"])

            counts = Counter(self.gesture_hist)
            best_gesture, best_votes = counts.most_common(1)[0]

            if best_gesture != "—" and best_votes >= cfg.gesture_min_votes:
                self.last_gesture = {
                    "gesture": best_gesture,
                    "gesture_conf": round(best_votes / len(self.gesture_hist), 3),
                }

    def _expire_gesture(self) -> None:
        """Clear gesture if hand has been missing too long."""
        if self.last_gesture["gesture"] == "—":
            return
        hand_gone_too_long = (
            self.last_hand_seen == 0.0
            or (time.time() - self.last_hand_seen) * 1000.0 > cfg.hand_lost_ms
        )
        if hand_gone_too_long:
            self.last_gesture = _EMPTY_GESTURE.copy()
            self.gesture_hist.clear()
            self.hand_miss_count = 0

    def _build_payload(self, latency_ms: float = 0.0) -> Dict[str, Any]:
        return {
            "person": self.last_face["person"],
            "face_conf": self.last_face["face_conf"],
            "distance": self.last_face.get("distance"),
            "gesture": self.last_gesture["gesture"],
            "gesture_conf": self.last_gesture["gesture_conf"],
            "latency_ms": round(latency_ms, 1),
            "ts": time.time(),
        }