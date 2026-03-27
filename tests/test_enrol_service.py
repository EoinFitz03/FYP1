"""
tests/test_enrol_service.py

Unit tests for EnrolService.
The Database is fully mocked so no SQLite file is touched.
"""

import sys
import os
import pytest
import numpy as np
from unittest.mock import MagicMock, patch, call

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "Demo1"))
sys.path.insert(0, os.path.join(ROOT, "backend"))

from services.enrol_service import EnrolService


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_service(min_ms: int = 0) -> EnrolService:
    """Return an EnrolService with a no-op on_saved callback."""
    on_saved = MagicMock()
    return EnrolService(
        db_path=":memory:",
        model="hog",
        min_ms_between_captures=min_ms,
        on_saved=on_saved,
    )


def _fake_encoding() -> np.ndarray:
    """Return a deterministic 128-d unit vector to stand in for a face encoding."""
    enc = np.ones(128, dtype=np.float64)
    return enc / np.linalg.norm(enc)


# ---------------------------------------------------------------------------
# start()
# ---------------------------------------------------------------------------

class TestEnrolServiceStart:

    def test_start_sets_active_true_with_valid_name(self):
        svc = _make_service()
        status = svc.start(name="Alice", num_samples=5)
        assert svc.active is True
        assert status["active"] is True

    def test_start_stores_name(self):
        svc = _make_service()
        svc.start(name="Bob", num_samples=5)
        assert svc.name == "Bob"

    def test_start_clamps_num_samples_minimum(self):
        svc = _make_service()
        svc.start(name="Alice", num_samples=1)   # below min of 3
        assert svc.target == 3

    def test_start_clamps_num_samples_maximum(self):
        svc = _make_service()
        svc.start(name="Alice", num_samples=999)  # above max of 30
        assert svc.target == 30

    def test_start_with_empty_name_leaves_inactive(self):
        svc = _make_service()
        status = svc.start(name="", num_samples=5)
        assert svc.active is False
        assert status["active"] is False
        assert status["error"] == "missing_name"

    def test_start_resets_collected(self):
        svc = _make_service()
        svc.collected = [_fake_encoding()]   # simulate leftover state
        svc.start(name="Alice", num_samples=5)
        assert svc.collected == []

    def test_start_returns_zero_captured(self):
        svc = _make_service()
        status = svc.start(name="Alice", num_samples=5)
        assert status["captured"] == 0

    def test_start_returns_correct_target(self):
        svc = _make_service()
        status = svc.start(name="Alice", num_samples=10)
        assert status["target"] == 10


# ---------------------------------------------------------------------------
# cancel()
# ---------------------------------------------------------------------------

class TestEnrolServiceCancel:

    def test_cancel_sets_active_false(self):
        svc = _make_service()
        svc.start(name="Alice", num_samples=5)
        svc.cancel()
        assert svc.active is False

    def test_cancel_clears_name(self):
        svc = _make_service()
        svc.start(name="Alice", num_samples=5)
        svc.cancel()
        assert svc.name == ""

    def test_cancel_clears_collected(self):
        svc = _make_service()
        svc.start(name="Alice", num_samples=5)
        svc.collected = [_fake_encoding()]
        svc.cancel()
        assert svc.collected == []

    def test_cancel_returns_error_cancelled(self):
        svc = _make_service()
        svc.start(name="Alice", num_samples=5)
        status = svc.cancel()
        assert status["error"] == "cancelled"

    def test_cancel_when_not_active_still_safe(self):
        svc = _make_service()
        status = svc.cancel()   # never started
        assert status["active"] is False


# ---------------------------------------------------------------------------
# try_capture() — inactive guard
# ---------------------------------------------------------------------------

class TestEnrolServiceTryCaptureInactive:

    def test_returns_none_when_not_active(self):
        svc = _make_service()
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = svc.try_capture(dummy_frame)
        assert result is None


# ---------------------------------------------------------------------------
# try_capture() — happy path (face found each frame)
# ---------------------------------------------------------------------------

class TestEnrolServiceTryCaptureHappyPath:

    @patch("services.enrol_service.face_recognition")
    @patch("services.enrol_service.Database")
    def test_accumulates_samples_until_target(self, MockDB, mock_fr):
        """
        Each call to try_capture should accumulate one encoding.
        After `target` captures the service marks itself done.
        """
        mock_fr.face_locations.return_value = [(0, 100, 100, 0)]
        mock_fr.face_encodings.return_value = [_fake_encoding()]

        mock_db_instance = MagicMock()
        mock_db_instance.get_user_id.return_value = None
        mock_db_instance.add_user.return_value = 1
        MockDB.return_value = mock_db_instance

        svc = _make_service(min_ms=0)
        svc.start(name="Alice", num_samples=3)

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        statuses = []
        for _ in range(3):
            result = svc.try_capture(dummy_frame)
            if result is not None:
                statuses.append(result)

        final = statuses[-1]
        assert final["done"] is True
        assert svc.active is False

    @patch("services.enrol_service.face_recognition")
    @patch("services.enrol_service.Database")
    def test_calls_on_saved_when_complete(self, MockDB, mock_fr):
        mock_fr.face_locations.return_value = [(0, 100, 100, 0)]
        mock_fr.face_encodings.return_value = [_fake_encoding()]

        mock_db_instance = MagicMock()
        mock_db_instance.get_user_id.return_value = 42
        MockDB.return_value = mock_db_instance

        on_saved = MagicMock()
        svc = EnrolService(
            db_path=":memory:",
            model="hog",
            min_ms_between_captures=0,
            on_saved=on_saved,
        )
        svc.start(name="Alice", num_samples=3)

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        for _ in range(3):
            svc.try_capture(dummy_frame)

        on_saved.assert_called_once()

    @patch("services.enrol_service.face_recognition")
    def test_returns_none_when_no_face_detected(self, mock_fr):
        mock_fr.face_locations.return_value = []   # no face in frame
        mock_fr.face_encodings.return_value = []

        svc = _make_service(min_ms=0)
        svc.start(name="Alice", num_samples=5)
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        result = svc.try_capture(dummy_frame)
        assert result is None

    @patch("services.enrol_service.face_recognition")
    def test_returns_none_when_multiple_faces_detected(self, mock_fr):
        """EnrolService only accepts frames with exactly one face."""
        mock_fr.face_locations.return_value = [(0, 100, 100, 0), (0, 200, 200, 100)]
        mock_fr.face_encodings.return_value = [_fake_encoding(), _fake_encoding()]

        svc = _make_service(min_ms=0)
        svc.start(name="Alice", num_samples=5)
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        result = svc.try_capture(dummy_frame)
        assert result is None

    @patch("services.enrol_service.face_recognition")
    @patch("services.enrol_service.Database")
    def test_intermediate_status_shows_progress(self, MockDB, mock_fr):
        mock_fr.face_locations.return_value = [(0, 100, 100, 0)]
        mock_fr.face_encodings.return_value = [_fake_encoding()]
        MockDB.return_value = MagicMock()

        svc = _make_service(min_ms=0)
        svc.start(name="Alice", num_samples=5)
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        result = svc.try_capture(dummy_frame)   # first capture (of 5)
        assert result is not None
        assert result["captured"] == 1
        assert result["done"] is False
        assert result["active"] is True