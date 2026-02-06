from __future__ import annotations

from typing import Any, Dict, Optional

import cv2
import numpy as np


class GestureService:
    """
    Same detect_gesture_fast logic as before.
    Keeps mediapipe init/close in one place.
    """

    def __init__(self, gesture_small_width: int = 320) -> None:
        self.gesture_small_width = gesture_small_width

        # Import gesture module 
        try:
            from gestures_live import (
                mp_hands,
                classify_gesture,
                Gesture,
                MIN_DET_CONF,
                MIN_TRK_CONF,
            )
            self.mp_hands = mp_hands
            self.classify_gesture = classify_gesture
            self.Gesture = Gesture
            self.MIN_DET_CONF = MIN_DET_CONF
            self.MIN_TRK_CONF = MIN_TRK_CONF
        except Exception as e:
            print(f"[backend] Gesture imports failed: {e}")
            self.mp_hands = None
            self.classify_gesture = None
            self.Gesture = None
            self.MIN_DET_CONF = 0.5
            self.MIN_TRK_CONF = 0.5

        self.hands_detector: Optional[Any] = None

    def startup(self) -> None:
        try:
            if self.mp_hands is not None:
                self.hands_detector = self.mp_hands.Hands(
                    model_complexity=0,
                    max_num_hands=1,
                    min_detection_confidence=self.MIN_DET_CONF,
                    min_tracking_confidence=self.MIN_TRK_CONF,
                )
                print("[backend] MediaPipe Hands initialised")
            else:
                self.hands_detector = None
                print("[backend] MediaPipe Hands not available (imports failed)")
        except Exception as e:
            self.hands_detector = None
            print(f"[backend] Failed to initialise MediaPipe Hands: {e}")

    def shutdown(self) -> None:
        try:
            if self.hands_detector is not None:
                self.hands_detector.close()
                self.hands_detector = None
        except Exception:
            pass

    def detect_gesture_fast(self, bgr: np.ndarray) -> Dict[str, Any]:
        """
        Unchanged behaviour:
        - resize for speed
        - mediapipe hands
        - classify gesture
        - return "—" if no hand / unknown
        """
        if self.hands_detector is None or self.classify_gesture is None:
            return {"gesture": "—", "gesture_conf": 0.0}

        h, w = bgr.shape[:2]
        if w > self.gesture_small_width:
            scale = self.gesture_small_width / float(w)
            small = cv2.resize(bgr, (self.gesture_small_width, int(h * scale)))
        else:
            small = bgr

        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        res = self.hands_detector.process(rgb)

        if not res.multi_hand_landmarks:
            return {"gesture": "—", "gesture_conf": 0.0}

        try:
            g = self.classify_gesture(res.multi_hand_landmarks[0])
            if self.Gesture is not None and g == self.Gesture.UNKNOWN:
                return {"gesture": "—", "gesture_conf": 0.0}

            label = str(g.value) if hasattr(g, "value") else str(g)
            return {"gesture": label, "gesture_conf": 1.0}
        except Exception:
            return {"gesture": "—", "gesture_conf": 0.0}
