from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np # numpy to average collected encodings
import face_recognition
# Import timing, image-processing, face-recognition, and database tools used during enrolment
from db import Database  # Demo1/db.py

class EnrolService: # Service responsible for collecting and saving new user face encodings

    def __init__( # Store enrolment settings and initialise the temporary state used while collecting samples (2)
        self,
        db_path: str,
        model: str,
        min_ms_between_captures: int,
        on_saved: Callable[[], None],
    ) -> None:
        self.db_path = db_path # Save database path, face model settings, capture timing, and post-save callback
        self.model = model
        self.min_ms_between_captures = min_ms_between_captures # wait time between samples
        self.on_saved = on_saved

        self.active: bool = False # Track whether enrolment is active, who is being enrolled, how many samples are needed, and what has been captured so far
        self.name: str = ""
        self.target: int = 10
        self.collected: List[np.ndarray] = [] # list of face encodings gathered so far
        self.last_capture: float = 0.0

    def start(self, name: str, num_samples: int) -> Dict[str, Any]:
        # Start a new enrolment session and reset any previous sample collection state
        self.name = str(name).strip() # Clean the provided user name before starting enrolment
        self.target = max(3, min(30, int(num_samples))) # Clamp the requested number of samples to a safe range between 3 and 30
        self.collected = [] 
        self.last_capture = 0.0
        self.active = bool(self.name) # Reset collected samples and activate enrolment only if a valid name was provided

        return { # Return the initial enrolment status so the frontend knows whether enrolment started successfully
            "active": self.active, #enrolment running or not
            "name": self.name, # user being enrolled
            "captured": 0, # current collected samples
            "target": self.target, # required total samples
            "done": False, 
            "error": None if self.active else "missing_name",
            "ts": time.time(), # timestamp
        }

    def cancel(self) -> Dict[str, Any]: # Cancel the current enrolment session and clear temporary enrolment data
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
        # Try to capture one enrolment sample from the current frame and save once the target count is reached
        """
        Called on each incoming frame while enrolment is active.
        Returns a status payload when something meaningful happens, else None.
        """
        if not self.active: # Ignore incoming frames when enrolment is not currently active
            return None

        if (time.time() - self.last_capture) * 1000.0 < self.min_ms_between_captures:
            return None # Enforce a minimum time gap so near-duplicate face samples are not captured too quickly

        enc = self._extract_single_face_encoding(bgr) 
        # Extract a single valid face encoding from the frame and skip the frame if this is not possible
        if enc is None:
            return None

        self.collected.append(enc) # Store the captured face encoding and record when the sample was taken
        self.last_capture = time.time()

        if len(self.collected) < self.target:
            return { # Return progress to the frontend while more enrolment samples are still needed
                "active": True,
                "name": self.name,
                "captured": len(self.collected),
                "target": self.target,
                "done": False,
                "error": None,
                "ts": time.time(),
            }

        # All samples collected — save and reset (7)
        # Once enough samples are collected, average them into one final encoding and save it to the database
        mean_encoding = np.mean(self.collected, axis=0) # Average the collected face encodings to create one stable enrolled encoding
        db = Database(self.db_path) # Open the face database so the enrolled user can be saved
        user_id = db.get_user_id(self.name) or db.add_user(self.name) # Reuse an existing user record if present, otherwise create a new user entry
        db.add_face_encoding(user_id, mean_encoding) # Save the averaged face encoding for the enrolled user
        db.close()

        self.on_saved() # Trigger post-save actions, such as reloading known faces into the recognition service
        saved_target = self.target

        self.active = False # Reset temporary enrolment state after the new user has been saved
        self.name = ""
        self.collected = []
        self.last_capture = 0.0

        return { # Return a final completion status so the frontend knows enrolment has finished successfully
            "active": False,
            "name": "",
            "captured": saved_target,
            "target": saved_target,
            "done": True,
            "error": None,
            "ts": time.time(),
        }

    def _extract_single_face_encoding(self, bgr: np.ndarray) -> Optional[np.ndarray]:
        # Extract exactly one face encoding from the current frame for enrolment
        small = cv2.resize(bgr, (0, 0), fx=0.25, fy=0.25)
        # Shrink the frame to speed up face detection during enrolment
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        # Convert the OpenCV frame from BGR to RGB for the face_recognition library
        locs = face_recognition.face_locations(rgb_small, model=self.model)
        # Detect face locations and skip the frame if no face is found
        if not locs:
            return None

        encs = face_recognition.face_encodings(rgb_small, locs)
        if not encs or len(encs) != 1:
            # Generate face encodings and accept the frame only when exactly one face is present
            return None

        return encs[0]