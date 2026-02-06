from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import time
import numpy as np
import cv2
import face_recognition

from db import Database  # Demo1/db.py 


class EnrolService:
    """
    Same enrol behaviour you already implemented:
    - enrol_start / enrol_cancel
    - collect N encodings from incoming frames
    - average encodings
    - save into Demo1 system.db
    - reload known faces (callback) so Live updates immediately
    """

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
        n = int(num_samples)
        self.target = max(3, min(30, n))

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

    def extract_single_face_encoding(self, bgr: np.ndarray) -> Optional[np.ndarray]:
        small = cv2.resize(bgr, (0, 0), fx=0.25, fy=0.25)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        locs = face_recognition.face_locations(rgb_small, model=self.model)
        if not locs:
            return None

        encs = face_recognition.face_encodings(rgb_small, locs)
        if not encs:
            return None

        if len(encs) != 1:
            return None

        return encs[0]

    def try_capture(self, bgr: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Called on each incoming frame.
        Returns an enrol_status payload dict when:
        - we capture a new encoding, OR
        - we finish and save.
        Otherwise returns None.
        """
        if not self.active:
            return None

        now = time.time()
        if (now - self.last_capture) * 1000.0 < self.min_ms_between_captures:
            return None

        enc = self.extract_single_face_encoding(bgr)
        if enc is None:
            return None

        self.collected.append(enc)
        self.last_capture = now

        # progress update
        progress = {
            "active": True,
            "name": self.name,
            "captured": len(self.collected),
            "target": self.target,
            "done": False,
            "error": None,
            "ts": time.time(),
        }

        if len(self.collected) < self.target:
            return progress

        # finish: average + save
        mean_encoding = np.mean(self.collected, axis=0)

        db = Database(self.db_path)
        user_id = db.get_user_id(self.name)
        if user_id is None:
            user_id = db.add_user(self.name)
        db.add_face_encoding(user_id, mean_encoding)
        db.close()

        # reload face encodings (same behaviour as before)
        self.on_saved()

        # reset enrol state
        self.active = False
        self.name = ""
        self.collected = []
        self.last_capture = 0.0

        return {
            "active": False,
            "name": "",
            "captured": self.target,
            "target": self.target,
            "done": True,
            "error": None,
            "ts": time.time(),
        }
