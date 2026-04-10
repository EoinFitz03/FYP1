# tests/test_gesture_service_backend.py

import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "backend"))

from services.gesture_service import GestureService


def _dummy_frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


def _make_result(landmarks=True, handedness_label="Right"):
    res = SimpleNamespace()
    res.multi_hand_landmarks = [MagicMock()] if landmarks else []
    if handedness_label is None:
        res.multi_handedness = []
    else:
        cls = SimpleNamespace(label=handedness_label)
        handed = SimpleNamespace(classification=[cls])
        res.multi_handedness = [handed]
    return res


def test_detect_gesture_fast_returns_default_when_detector_missing():
    svc = GestureService()
    svc.hands_detector = None
    svc.classify_gesture = MagicMock()

    result = svc.detect_gesture_fast(_dummy_frame())

    assert result == {"gesture": "—", "gesture_conf": 0.0}


@patch("services.gesture_service.cv2.cvtColor", side_effect=lambda img, code: img)
def test_detect_gesture_fast_returns_default_when_no_landmarks(mock_cvt):
    svc = GestureService()
    svc.hands_detector = MagicMock()
    svc.classify_gesture = MagicMock()
    svc.hands_detector.process.return_value = _make_result(landmarks=False)

    result = svc.detect_gesture_fast(_dummy_frame())

    assert result == {"gesture": "—", "gesture_conf": 0.0}


@patch("services.gesture_service.cv2.cvtColor", side_effect=lambda img, code: img)
def test_detect_gesture_fast_returns_label_for_known_gesture(mock_cvt):
    svc = GestureService()
    svc.hands_detector = MagicMock()
    svc.Gesture = SimpleNamespace(UNKNOWN="UNKNOWN")
    svc.classify_gesture = MagicMock(return_value=SimpleNamespace(value="Open Palm"))
    svc.hands_detector.process.return_value = _make_result(landmarks=True)

    result = svc.detect_gesture_fast(_dummy_frame())

    assert result["gesture"] == "Open Palm"
    assert result["gesture_conf"] == 1.0


@patch("services.gesture_service.cv2.cvtColor", side_effect=lambda img, code: img)
def test_detect_gesture_fast_returns_default_when_classifier_raises(mock_cvt):
    svc = GestureService()
    svc.hands_detector = MagicMock()
    svc.classify_gesture = MagicMock(side_effect=Exception("boom"))
    svc.hands_detector.process.return_value = _make_result(landmarks=True)

    result = svc.detect_gesture_fast(_dummy_frame())

    assert result == {"gesture": "—", "gesture_conf": 0.0}


@patch("services.gesture_service.cv2.cvtColor", side_effect=lambda img, code: img)
def test_predict_trained_gesture_returns_default_when_predictor_not_ready(mock_cvt):
    svc = GestureService()
    svc.hands_detector = MagicMock()
    svc.predictor = MagicMock()
    svc.predictor.is_ready = False

    result = svc.predict_trained_gesture(_dummy_frame())

    assert result == {"gesture": "—", "gesture_conf": 0.0}


@patch("services.gesture_service.cv2.cvtColor", side_effect=lambda img, code: img)
def test_predict_trained_gesture_returns_prediction(mock_cvt):
    svc = GestureService()
    svc.hands_detector = MagicMock()
    svc.predictor = MagicMock()
    svc.predictor.is_ready = True
    svc.hands_detector.process.return_value = _make_result(landmarks=True, handedness_label="Left")
    fake_landmarks = MagicMock()
    svc.hands_detector.process.return_value.multi_hand_landmarks[0].landmark = fake_landmarks
    svc.predictor.predict_from_mediapipe.return_value = ("Thumbs Up", 0.91)

    result = svc.predict_trained_gesture(_dummy_frame())

    assert result["gesture"] == "Thumbs Up"
    assert result["gesture_conf"] == 0.91