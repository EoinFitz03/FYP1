"""
tests/test_face_service.py

Unit tests for FaceService.
face_recognition and Database are mocked so no real images or DB needed.
"""

import sys
import os
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "Demo1"))
sys.path.insert(0, os.path.join(ROOT, "backend"))

from services.face_service import FaceService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_service(tolerance: float = 0.6) -> FaceService:
    return FaceService(
        db_path=":memory:",
        tolerance=tolerance,
        downscale=1.0,
        model="hog",
    )


def _unit_vec(seed: int = 0, size: int = 128) -> np.ndarray:
    """Deterministic unit vector for use as a face encoding."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(size)
    return v / np.linalg.norm(v)


# ---------------------------------------------------------------------------
# load_known_faces()
# ---------------------------------------------------------------------------

class TestLoadKnownFaces:

    @patch("services.face_service.Database")
    def test_loads_encodings_and_names(self, MockDB):
        enc = _unit_vec(0)
        MockDB.return_value.load_all_encodings.return_value = ([enc], ["Alice"])

        svc = _make_service()
        svc.load_known_faces()

        assert len(svc.known_encodings) == 1
        assert svc.known_names == ["Alice"]

    @patch("services.face_service.Database")
    def test_empty_db_gives_empty_lists(self, MockDB):
        MockDB.return_value.load_all_encodings.return_value = ([], [])

        svc = _make_service()
        svc.load_known_faces()

        assert svc.known_encodings == []
        assert svc.known_names == []

    @patch("services.face_service.Database")
    def test_handles_db_exception_gracefully(self, MockDB):
        MockDB.return_value.load_all_encodings.side_effect = Exception("DB error")

        svc = _make_service()
        svc.load_known_faces()   # should not raise

        assert svc.known_encodings == []
        assert svc.known_names == []


# ---------------------------------------------------------------------------
# recognize_person() — no known faces loaded
# ---------------------------------------------------------------------------

class TestRecognizePersonNoKnownFaces:

    def test_returns_unknown_when_no_encodings_loaded(self):
        svc = _make_service()
        # known_encodings is empty by default
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = svc.recognize_person(dummy_frame)
        assert result["person"] == "Unknown"
        assert result["face_conf"] == 0.0

    @patch("services.face_service.face_recognition")
    def test_returns_unknown_when_no_face_in_frame(self, mock_fr):
        mock_fr.face_locations.return_value = []
        mock_fr.face_encodings.return_value = []

        svc = _make_service()
        svc.known_encodings = [_unit_vec(0)]
        svc.known_names = ["Alice"]

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = svc.recognize_person(dummy_frame)
        assert result["person"] == "Unknown"
        assert result["face_conf"] == 0.0


# ---------------------------------------------------------------------------
# recognize_person() — matching logic
# ---------------------------------------------------------------------------

class TestRecognizePersonMatching:

    @patch("services.face_service.face_recognition")
    def test_identifies_known_person_within_tolerance(self, mock_fr):
        known_enc = _unit_vec(42)
        # Return the same encoding — distance will be 0.0
        mock_fr.face_locations.return_value = [(0, 100, 100, 0)]
        mock_fr.face_encodings.return_value = [known_enc]
        mock_fr.face_distance.return_value = np.array([0.0])

        svc = _make_service(tolerance=0.6)
        svc.known_encodings = [known_enc]
        svc.known_names = ["Alice"]

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = svc.recognize_person(dummy_frame)

        assert result["person"] == "Alice"
        assert result["face_conf"] > 0.0

    @patch("services.face_service.face_recognition")
    def test_returns_unknown_when_distance_exceeds_tolerance(self, mock_fr):
        known_enc = _unit_vec(1)
        mock_fr.face_locations.return_value = [(0, 100, 100, 0)]
        mock_fr.face_encodings.return_value = [_unit_vec(2)]
        # Distance greater than tolerance → should return Unknown
        mock_fr.face_distance.return_value = np.array([0.9])

        svc = _make_service(tolerance=0.6)
        svc.known_encodings = [known_enc]
        svc.known_names = ["Alice"]

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = svc.recognize_person(dummy_frame)

        assert result["person"] == "Unknown"

    @patch("services.face_service.face_recognition")
    def test_picks_closest_match_from_multiple_known_people(self, mock_fr):
        enc_alice = _unit_vec(10)
        enc_bob   = _unit_vec(20)

        mock_fr.face_locations.return_value = [(0, 100, 100, 0)]
        mock_fr.face_encodings.return_value = [_unit_vec(10)]
        # Alice distance 0.1, Bob distance 0.5 → Alice should win
        mock_fr.face_distance.return_value = np.array([0.1, 0.5])

        svc = _make_service(tolerance=0.6)
        svc.known_encodings = [enc_alice, enc_bob]
        svc.known_names = ["Alice", "Bob"]

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = svc.recognize_person(dummy_frame)

        assert result["person"] == "Alice"

    @patch("services.face_service.face_recognition")
    def test_confidence_is_between_0_and_1(self, mock_fr):
        known_enc = _unit_vec(0)
        mock_fr.face_locations.return_value = [(0, 100, 100, 0)]
        mock_fr.face_encodings.return_value = [known_enc]
        mock_fr.face_distance.return_value = np.array([0.3])

        svc = _make_service(tolerance=0.6)
        svc.known_encodings = [known_enc]
        svc.known_names = ["Alice"]

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = svc.recognize_person(dummy_frame)

        assert 0.0 <= result["face_conf"] <= 1.0

    @patch("services.face_service.face_recognition")
    def test_distance_is_included_in_result(self, mock_fr):
        known_enc = _unit_vec(0)
        mock_fr.face_locations.return_value = [(0, 100, 100, 0)]
        mock_fr.face_encodings.return_value = [known_enc]
        mock_fr.face_distance.return_value = np.array([0.25])

        svc = _make_service(tolerance=0.6)
        svc.known_encodings = [known_enc]
        svc.known_names = ["Alice"]

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = svc.recognize_person(dummy_frame)

        assert result["distance"] is not None
        assert abs(result["distance"] - 0.25) < 0.001

    @patch("services.face_service.face_recognition")
    def test_exact_tolerance_boundary_is_accepted(self, mock_fr):
        """Distance exactly equal to tolerance should still be recognised."""
        known_enc = _unit_vec(0)
        mock_fr.face_locations.return_value = [(0, 100, 100, 0)]
        mock_fr.face_encodings.return_value = [known_enc]
        mock_fr.face_distance.return_value = np.array([0.6])   # == tolerance

        svc = _make_service(tolerance=0.6)
        svc.known_encodings = [known_enc]
        svc.known_names = ["Alice"]

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = svc.recognize_person(dummy_frame)

        assert result["person"] == "Alice"