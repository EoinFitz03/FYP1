from __future__ import annotations

from typing import Any, Dict, Optional

import cv2
import numpy as np

from training.predictor import GesturePredictor
# Import image-processing, MediaPipe, model-loading, and feature-extraction tools used for gesture recognition

class GestureService:
    """
    Backend service for live gesture recognition.

    Initialises MediaPipe Hands, supports trained-model prediction,
    and provides a rule-based fallback detector.
    """

    def __init__(self, gesture_small_width: int = 320) -> None:
        self.gesture_small_width = gesture_small_width # Store gesture-recognition settings and prepare runtime objects for MediaPipe and the trained model

        # Import gesture module
        try:
            from gestures_live import ( # Try to import the existing live gesture module and its MediaPipe / rule-based helpers
                mp_hands, #MediaPipe Hands module
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

        self.hands_detector: Optional[Any] = None # Placeholder for the MediaPipe Hands detector, initialised during startup

        # trained model predictor (loads backend/models/gesture_model.pkl if present)
        self.predictor = GesturePredictor() # Create the trained gesture predictor, which loads the saved model if available

    def startup(self) -> None: # Initialise the MediaPipe Hands detector when the backend starts
        try:
            if self.mp_hands is not None: # Create a lightweight single-hand detector for live gesture recognition
                self.hands_detector = self.mp_hands.Hands(
                    model_complexity=0, # means lighter / faster model
                    max_num_hands=1, # means only one hand is processed
                    min_detection_confidence=self.MIN_DET_CONF,
                    min_tracking_confidence=self.MIN_TRK_CONF,
                )
                print("[backend] MediaPipe Hands initialised") # Load the saved gesture model and label encoder if the training artefacts exist
            else: # Leave the detector disabled if gesture imports were not available
                self.hands_detector = None
                print("[backend] MediaPipe Hands not available (imports failed)")
        except Exception as e:
            self.hands_detector = None
            print(f"[backend] Failed to initialise MediaPipe Hands: {e}")

    def shutdown(self) -> None:
        try: # Close the MediaPipe hand detector and release resources during shutdown
            if self.hands_detector is not None:
                self.hands_detector.close()
                self.hands_detector = None
        except Exception:
            pass

    def detect_gesture_fast(self, bgr: np.ndarray) -> Dict[str, Any]: # Close the MediaPipe hand detector and release resources during shutdown
        """
        Unchanged behaviour:
        - resize for speed
        - mediapipe hands
        - classify gesture
        - return "—" if no hand / unknown
        """
        if self.hands_detector is None or self.classify_gesture is None:
            return {"gesture": "—", "gesture_conf": 0.0}

        h, w = bgr.shape[:2] # Resize large frames to a smaller width to reduce gesture-processing cost
        if w > self.gesture_small_width:
            scale = self.gesture_small_width / float(w)
            small = cv2.resize(bgr, (self.gesture_small_width, int(h * scale)))
        else:
            small = bgr

        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB) # Convert the OpenCV frame from BGR to RGB before MediaPipe processing
        res = self.hands_detector.process(rgb) # Run MediaPipe Hands to detect hand landmarks in the frame

        if not res.multi_hand_landmarks:
            return {"gesture": "—", "gesture_conf": 0.0}

        try: # Classify the first detected hand using the older rule-based gesture logic
            g = self.classify_gesture(res.multi_hand_landmarks[0])
            if self.Gesture is not None and g == self.Gesture.UNKNOWN:
                return {"gesture": "—", "gesture_conf": 0.0} # Return an empty result if the rule-based classifier reports an unknown gesture

            # Convert the detected gesture into a frontend-friendly label and return it
            label = str(g.value) if hasattr(g, "value") else str(g)
            return {"gesture": label, "gesture_conf": 1.0}
        except Exception:
            return {"gesture": "—", "gesture_conf": 0.0}

    def predict_trained_gesture(self, bgr: np.ndarray) -> Dict[str, Any]:
        # Predict a gesture using the trained model and return the same output format as the fallback detector
        """Predict using your trained Random Forest model.

        Returns a dict with the same keys as detect_gesture_fast:
          {"gesture": <label or "—">, "gesture_conf": <0..1>}
        """
        if self.hands_detector is None or self.predictor is None or not self.predictor.is_ready:
            return {"gesture": "—", "gesture_conf": 0.0} # Predict a gesture using the trained model and return the same output format as the fallback detector

        h, w = bgr.shape[:2] # Resize large frames before trained gesture prediction to keep processing fast
        if w > self.gesture_small_width:
            scale = self.gesture_small_width / float(w)
            small = cv2.resize(bgr, (self.gesture_small_width, int(h * scale)))
        else:
            small = bgr

        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB) # Convert to RGB and run MediaPipe Hands to extract hand landmarks
        res = self.hands_detector.process(rgb)

        if not res.multi_hand_landmarks:
            return {"gesture": "—", "gesture_conf": 0.0} # Return an empty result if no hand landmarks are available for trained prediction

        hand_side = "Unknown" # Return an empty result if no hand landmarks are available for trained prediction
        if res.multi_handedness and len(res.multi_handedness) > 0:
            hand_side = res.multi_handedness[0].classification[0].label  # Left/Right

        pred = self.predictor.predict_from_mediapipe( # Pass hand side and MediaPipe landmarks into the trained gesture predictor
            hand_side,
            res.multi_hand_landmarks[0].landmark,
        )
        if not pred:
            return {"gesture": "—", "gesture_conf": 0.0} # Return an empty result if the trained predictor cannot produce a gesture

        label, conf = pred
        return {"gesture": str(label), "gesture_conf": float(conf)}