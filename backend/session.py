"""
Per-connection session state.

Owns all the stateful logic that used to live inside ws_endpoint:
  - frame counter
  - face staleness / last-seen tracking
  - gesture smoothing / voting
  - hand-miss counter
"""
from __future__ import annotations

# Import time, history utilities, configuration, and service types used for session processing
import time
from collections import Counter, deque #deque stores recent gesture history,Counter helps vote for the most common gesture
from typing import Any, Dict

import numpy as np

from config import cfg
from services.face_service import FaceService
from services.gesture_service import GestureService

# Default values used when no face or gesture is currently recognised
_UNKNOWN_FACE: Dict[str, Any] = {"person": "Unknown", "face_conf": 0.0, "distance": None}
_EMPTY_GESTURE: Dict[str, Any] = {"gesture": "—", "gesture_conf": 0.0}


class SessionState: # Stores live recognition state for a single WebSocket client session
    """One instance per WebSocket connection."""
    # Count how many frames have been processed in this session
    def __init__(self) -> None:
        self.frame_i: int = 0

        # Face state,      Store the most recent face result and when a valid face was last seen
        self.last_face: Dict[str, Any] = _UNKNOWN_FACE.copy()
        self.last_face_seen: float = 0.0

        # Gesture state,     # Store the latest gesture result, recent gesture history, and hand-loss tracking
        self.last_gesture: Dict[str, Any] = _EMPTY_GESTURE.copy() # latest gesture result
        self.gesture_hist: deque = deque(maxlen=cfg.gesture_smooth_window) # recent gesture history, maxlen=cfg.gesture_smooth_window means only recent labels are kept
        self.last_hand_seen: float = 0.0
        self.hand_miss_count: int = 0 # helps avoid clearing gestures too aggressively

#(7)
# Process one frame, update face and gesture state, and return a payload for the frontend
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
        self.frame_i += 1     # Increase the session frame counter

        if self.frame_i % cfg.face_every_n_frames == 0: # Run face recognition only on configured frame intervals to reduce processing cost
            self._update_face(bgr, face_svc)

        if self.frame_i % cfg.gesture_every_n_frames == 0: # Run gesture recognition only on configured frame intervals
            self._update_gesture(bgr, gesture_svc)

        self._expire_gesture()    # Clear old gesture results if the hand has been missing for too long

        return self._build_payload()     # Return the latest combined face and gesture state as a frontend payload

    def build_snapshot(self) -> Dict[str, Any]: # Return the current session result without processing a new frame (8)
        """Return current state without running any detection (for non-frame messages)."""
        return self._build_payload(latency_ms=0)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _update_face(self, bgr: np.ndarray, face_svc: FaceService) -> None:# Update the stored face result using the current frame
        new_face = face_svc.recognize_person(bgr)     # Ask the face service to recognise the person in this frame

        if new_face["person"] != "Unknown": # If a known person is detected, store the result and update the last-seen timestamp
            self.last_face = new_face
            self.last_face_seen = time.time()
        else:         # Only reset to Unknown if the previous face has been missing longer than the configured timeout
            still_within_grace = ( # Keep the previous recognised face for a short time to avoid flicker when detection briefly fails
                self.last_face["person"] != "Unknown"
                and (time.time() - self.last_face_seen) * 1000.0 <= cfg.face_lost_ms
            )
            if not still_within_grace:
                self.last_face = new_face
# Update the stored gesture result using trained prediction first, then fallback detection
    def _update_gesture(self, bgr: np.ndarray, gesture_svc: GestureService) -> None:
        # Step 3: try trained gesture model first, then fall back to rule-based classifier
        raw = gesture_svc.predict_trained_gesture(bgr)
        if raw.get("gesture") == "—":
            raw = gesture_svc.detect_gesture_fast(bgr)

        if raw["gesture"] == "—":  # If no gesture is detected, count the miss and clear old gesture state only after repeated misses
            self.hand_miss_count += 1 # avoids instantly removing gesture 
            if self.hand_miss_count >= cfg.hand_miss_clear_count:
                self.last_gesture = _EMPTY_GESTURE.copy()
                self.gesture_hist.clear()
        else:     # If a gesture is detected, reset miss tracking and add the gesture to the smoothing history
            self.hand_miss_count = 0 # reset miss counter 
            self.last_hand_seen = time.time()
            self.gesture_hist.append(raw["gesture"])
            # Count recent gesture predictions and choose the most common one
            counts = Counter(self.gesture_hist) #This is the actual smoothing logic.
            best_gesture, best_votes = counts.most_common(1)[0] # It uses recent frames and votes
             # Update the final gesture only if it has enough votes in the recent history window
            if best_gesture != "—" and best_votes >= cfg.gesture_min_votes:
                self.last_gesture = {
                    "gesture": best_gesture, # Confidence is based on how often the winning gesture appears in the smoothing window
                    "gesture_conf": round(best_votes / len(self.gesture_hist), 3),
                }

    def _expire_gesture(self) -> None: # Remove stale gesture results after the hand has been absent longer than the timeout (12)
        """Clear gesture if hand has been missing too long."""
        if self.last_gesture["gesture"] == "—":     # Do nothing if there is no active gesture stored
            return
        hand_gone_too_long = (     # Check whether the hand has been absent longer than the configured gesture timeout
            self.last_hand_seen == 0.0
            or (time.time() - self.last_hand_seen) * 1000.0 > cfg.hand_lost_ms
        )
        if hand_gone_too_long:     # Clear the stored gesture and reset smoothing state when the hand is gone too long
            self.last_gesture = _EMPTY_GESTURE.copy()
            self.gesture_hist.clear()
            self.hand_miss_count = 0

    def _build_payload(self, latency_ms: float = 0.0) -> Dict[str, Any]:
        return { # Build the final response payload from the current session state
            "person": self.last_face["person"],
            "face_conf": self.last_face["face_conf"],
            "distance": self.last_face.get("distance"),
            "gesture": self.last_gesture["gesture"],
            "gesture_conf": self.last_gesture["gesture_conf"],
            "latency_ms": round(latency_ms, 1),
            "ts": time.time(),
        }