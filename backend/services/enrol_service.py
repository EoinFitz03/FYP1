from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np
import face_recognition

from db import Database  # Demo1/db.py

class EnrolService:

    def __init__(
        self,
        db_path: str,
        model: str,
        min_ms_between_captures: int,
        on_saved: Callable[[], None],
    ) -> None:
        self.db_path = db_path
        self.model = model
        self.min_ms_between_captures = min_ms_between_captures
        self.on_saved = on_saved

        self.active: bool = False
        self.name: str = ""
        self.target: int = 10
        self.collected: List[np.ndarray] = []
        self.last_capture: float = 0.0

    def start(self, name: str, num_samples: int) -> Dict[str, Any]:
        self.name = str(name).strip()
        self.target = max(3, min(30, int(num_samples)))
        self.collected = []
        self.last_capture = 0.0
        self.active = bool(self.name)

        return {
            "active": self.active,
            "name": self.name,
            "captured": 0,
            "target": self.target,
            "done": False,
            "error": None if self.active else "missing_name",
            "ts": time.time(),
        }

    def cancel(self) -> Dict[str, Any]:
        self.active = False
        self.name = ""
        self.collected = []
        self.last_capture = 0.0

        return {
            "active": False,
            "name": "",
            "captured": 0,
            "target": 0,
            "done": False,
            "error": "cancelled",
            "ts": time.time(),
        }

    def try_capture(self, bgr: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Called on each incoming frame while enrolment is active.
        Returns a status payload when something meaningful happens, else None.
        """
        if not self.active:
            return None

        if (time.time() - self.last_capture) * 1000.0 < self.min_ms_between_captures:
            return None

        enc = self._extract_single_face_encoding(bgr)
        if enc is None:
            return None

        self.collected.append(enc)
        self.last_capture = time.time()

        if len(self.collected) < self.target:
            return {
                "active": True,
                "name": self.name,
                "captured": len(self.collected),
                "target": self.target,
                "done": False,
                "error": None,
                "ts": time.time(),
            }

        # All samples collected — save and reset
        mean_encoding = np.mean(self.collected, axis=0)
        db = Database(self.db_path)
        user_id = db.get_user_id(self.name) or db.add_user(self.name)
        db.add_face_encoding(user_id, mean_encoding)
        db.close()

        self.on_saved()
        saved_target = self.target

        self.active = False
        self.name = ""
        self.collected = []
        self.last_capture = 0.0

        return {
            "active": False,
            "name": "",
            "captured": saved_target,
            "target": saved_target,
            "done": True,
            "error": None,
            "ts": time.time(),
        }

    def _extract_single_face_encoding(self, bgr: np.ndarray) -> Optional[np.ndarray]:
        small = cv2.resize(bgr, (0, 0), fx=0.25, fy=0.25)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        locs = face_recognition.face_locations(rgb_small, model=self.model)
        if not locs:
            return None

        encs = face_recognition.face_encodings(rgb_small, locs)
        if not encs or len(encs) != 1:
            return None

        return encs[0]