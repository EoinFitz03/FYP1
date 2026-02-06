import base64
from typing import Optional

import cv2
import numpy as np


def decode_base64_jpeg(base64_jpeg: str) -> Optional[np.ndarray]:
    """
    Unchanged behaviour:
    base64 JPEG -> numpy BGR image or None if decode fails.
    """
    try:
        jpg_bytes = base64.b64decode(base64_jpeg)
        arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None
