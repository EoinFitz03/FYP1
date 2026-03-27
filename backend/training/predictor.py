# backend/training/predictor.py
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from training.model_io import load_gesture_model


class GesturePredictor:
    """Predict gestures from MediaPipe hand landmarks using the trained model."""

    def __init__(self, model_path: Optional[str] = None) -> None:
        bundle = load_gesture_model(model_path) if model_path else load_gesture_model()
        self.model = None
        self.feature_columns = None
        self.labels = None

        if bundle:
            self.model = bundle.get("model")
            self.feature_columns = bundle.get("feature_columns")
            self.labels = bundle.get("labels")

    @property
    def is_ready(self) -> bool:
        return self.model is not None and self.feature_columns is not None

    def _hand_enc(self, hand_side: str) -> int:
        # MUST match trainer.py mapping
        hand_map = {"Left": 0, "Right": 1, "Unknown": 2}
        return int(hand_map.get(hand_side, 2))

    def predict_from_mediapipe(
        self,
        hand_side: str,
        landmarks,
    ) -> Optional[Tuple[str, float]]:
        """Return (label, confidence) or None."""

        if not self.is_ready:
            return None

        lms = list(landmarks)
        if len(lms) != 21:
            return None

        feats = {}

        # Optional feature if included during training
        if "hand_enc" in self.feature_columns:
            feats["hand_enc"] = self._hand_enc(hand_side)

        for i, lm in enumerate(lms):
            feats[f"x{i}"] = float(lm.x)
            feats[f"y{i}"] = float(lm.y)
            feats[f"z{i}"] = float(lm.z)

        X = np.array(
            [[feats.get(c, 0.0) for c in self.feature_columns]],
            dtype=np.float32,
        )

        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)[0]
            best_idx = int(np.argmax(proba))
            label = str(self.model.classes_[best_idx])
            conf = float(proba[best_idx])
            return label, conf

        label = str(self.model.predict(X)[0])
        return label, 1.0