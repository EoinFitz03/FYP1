from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import cv2
import face_recognition

from db import Database  # Demo1/db.py 


class FaceService:
    """
    Same logic as your original recognize_person + DB loading.
    Just wrapped in a class so app.py stays small.
    """

    def __init__(self, db_path: str, tolerance: float, downscale: float, model: str) -> None:
        self.db_path = db_path
        self.tolerance = tolerance
        self.downscale = downscale
        self.model = model

        self.known_encodings: List[np.ndarray] = []
        self.known_names: List[str] = []

    def load_known_faces_from_db(self) -> Tuple[List[np.ndarray], List[str]]:
        db = Database(self.db_path)
        encs, names = db.load_all_encodings()
        encs = [np.asarray(e) for e in encs]
        names = [str(n) for n in names]
        return encs, names

    def load_known_faces(self) -> None:
        try:
            self.known_encodings, self.known_names = self.load_known_faces_from_db()
            print(f"[backend] Loaded {len(self.known_encodings)} encodings from {self.db_path}")
        except Exception as e:
            print(f"[backend] Failed loading encodings: {e}")
            self.known_encodings, self.known_names = [], []

    def recognize_person(self, bgr: np.ndarray) -> Dict[str, Any]:
        if len(self.known_encodings) == 0:
            return {"person": "Unknown", "face_conf": 0.0, "distance": None}

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        if self.downscale != 1.0:
            rgb_small = cv2.resize(rgb, (0, 0), fx=self.downscale, fy=self.downscale)
        else:
            rgb_small = rgb

        locations = face_recognition.face_locations(rgb_small, model=self.model)
        if not locations:
            return {"person": "Unknown", "face_conf": 0.0, "distance": None}

        encs = face_recognition.face_encodings(rgb_small, locations)
        if not encs:
            return {"person": "Unknown", "face_conf": 0.0, "distance": None}

        best_name = "Unknown"
        best_dist = 999.0

        for enc in encs:
            distances = face_recognition.face_distance(self.known_encodings, enc)
            if len(distances) == 0:
                continue
            i = int(np.argmin(distances))
            d = float(distances[i])
            if d < best_dist:
                best_dist = d
                best_name = self.known_names[i] if d <= self.tolerance else "Unknown"

        if best_dist == 999.0:
            return {"person": "Unknown", "face_conf": 0.0, "distance": None}

        conf = max(0.0, min(1.0, 1.0 - (best_dist / self.tolerance)))
        return {"person": best_name, "face_conf": round(conf, 3), "distance": round(best_dist, 4)}
