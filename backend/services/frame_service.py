import base64 # decodes the frontend text image data
from typing import Optional

import cv2 # holds the raw byte buffer
import numpy as np # holds the raw byte buffer


def decode_base64_jpeg(base64_jpeg: str) -> Optional[np.ndarray]:
     # Decode a base64 JPEG string from the frontend into an OpenCV BGR image
    """
    Unchanged behaviour:
    base64 JPEG -> numpy BGR image or None if decode fails.
    acts as a Bridge  
    converts incomign frame data inot openCV image format needed for face and gesture recognition
    """
    try:
        jpg_bytes = base64.b64decode(base64_jpeg) # Convert the base64 text into raw JPEG bytes
        arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None
