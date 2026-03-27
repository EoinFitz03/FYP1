"""
conftest.py — project-wide pytest configuration.

Stubs out heavy runtime imports (mediapipe, cv2, face_recognition)
so the test suite runs without a GPU, camera, or full ML stack installed.
The actual logic under test is pure Python and doesn't need them.
"""

import sys
from unittest.mock import MagicMock

# Stub mediapipe before any module imports it
mp_stub = MagicMock()
mp_stub.solutions.hands = MagicMock()
mp_stub.solutions.drawing_utils = MagicMock()
mp_stub.solutions.drawing_styles = MagicMock()
sys.modules.setdefault("mediapipe", mp_stub)

# Stub cv2
sys.modules.setdefault("cv2", MagicMock())

# Stub face_recognition (individual tests patch it where needed)
sys.modules.setdefault("face_recognition", MagicMock())