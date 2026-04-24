from __future__ import annotations
from typing import Any, Dict, List, Tuple

import numpy as np
import cv2 # used to convert BGR to RGB and resize frames
import face_recognition # does face detection and encoding

from db import Database  # Demo1/db.py
# Import typing, image-processing, face-recognition, and database tools used for face matching

class FaceService:  # # Handles loading known faces and recognising people from live frames

# Store face-recognition settings and prepare in-memory lists for known encodings and names (3)
    def __init__(self, db_path: str, tolerance: float, downscale: float, model: str) -> None:
        self.db_path = db_path
        self.tolerance = tolerance # matching threshold
        self.downscale = downscale # resize factor for faster processing
        self.model = model 

        self.known_encodings: List[np.ndarray] = []
        self.known_names: List[str] = []

    def load_known_faces(self) -> None: # Load all saved face encodings and names from the database into memory
        try:
            db = Database(self.db_path) # Open the face database
            encs, names = db.load_all_encodings()
            db.close() # Close the database once the data has been loaded
            self.known_encodings = [np.asarray(e) for e in encs]
            self.known_names = [str(n) for n in names] # Convert loaded encodings and names into in-memory lists used during matching
            print(f"[FaceService] Loaded {len(self.known_encodings)} encodings from {self.db_path}")
        except Exception as e:
            print(f"[FaceService] Failed loading encodings: {e}")
            self.known_encodings, self.known_names = [], [] # Fail safely by clearing in-memory face data if loading from the database does not work

    def recognize_person(self, bgr: np.ndarray) -> Dict[str, Any]: # (5)Recognise the best matching known person from a single BGR frame
        if not self.known_encodings:
            return {"person": "Unknown", "face_conf": 0.0, "distance": None}
            # Return Unknown if there are no loaded face encodings to compare against
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) # Convert OpenCV BGR image to RGB because the face_recognition library expects RGB input
        # Convert OpenCV BGR image to RGB because the face_recognition library expects RGB input
        rgb_small = (
            cv2.resize(rgb, (0, 0), fx=self.downscale, fy=self.downscale)
            if self.downscale != 1.0
            else rgb 
        ) # Optionally shrink the frame to speed up face detection and encoding

        locations = face_recognition.face_locations(rgb_small, model=self.model)
        # Detect face locations in the frame using the configured face-detection model (9)
        if not locations: 
            return {"person": "Unknown", "face_conf": 0.0, "distance": None}

        encs = face_recognition.face_encodings(rgb_small, locations) 
        if not encs: # Generate face encodings for each detected face location (10)
            return {"person": "Unknown", "face_conf": 0.0, "distance": None}

        best_name = "Unknown" # Start with a default Unknown result and a very large best-match distance
        best_dist = 999.0 

        for enc in encs:
            distances = face_recognition.face_distance(self.known_encodings, enc) # Compute similarity distances between this detected face and all known faces
            if not len(distances):
                continue
            i = int(np.argmin(distances)) # Find the known face with the smallest distance
            d = float(distances[i]) # Read the closest face distance as a numeric value
            if d < best_dist: # Update the stored best match if this face is closer than previous candidates
                best_dist = d
                best_name = self.known_names[i] if d <= self.tolerance else "Unknown"

        if best_dist == 999.0:
            return {"person": "Unknown", "face_conf": 0.0, "distance": None}

        conf = max(0.0, min(1.0, 1.0 - (best_dist / self.tolerance)))
        # Convert the best distance into a simple bounded confidence score between 0 and 1
        return {"person": best_name, "face_conf": round(conf, 3), "distance": round(best_dist, 4)}