# tests/test_frame_service.py

import os
import sys
from unittest.mock import patch
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "backend"))

from services.frame_service import decode_base64_jpeg


@patch("services.frame_service.cv2.imdecode")
@patch("services.frame_service.base64.b64decode")
def test_decode_base64_jpeg_returns_image_when_valid(mock_b64decode, mock_imdecode):
    dummy = np.zeros((10, 10, 3), dtype=np.uint8)
    mock_b64decode.return_value = b"fakejpegbytes"
    mock_imdecode.return_value = dummy

    result = decode_base64_jpeg("abc123")

    assert result is dummy


@patch("services.frame_service.cv2.imdecode")
@patch("services.frame_service.base64.b64decode")
def test_decode_base64_jpeg_returns_none_when_imdecode_fails(mock_b64decode, mock_imdecode):
    mock_b64decode.return_value = b"fakejpegbytes"
    mock_imdecode.return_value = None

    result = decode_base64_jpeg("abc123")

    assert result is None


@patch("services.frame_service.base64.b64decode")
def test_decode_base64_jpeg_returns_none_when_base64_decode_raises(mock_b64decode):
    mock_b64decode.side_effect = Exception("bad base64")

    result = decode_base64_jpeg("not-valid")

    assert result is None