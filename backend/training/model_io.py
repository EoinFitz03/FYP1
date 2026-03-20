# backend/training/model_io.py
import os
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # .../backend
MODEL_PATH = os.path.join(BASE_DIR, "models", "gesture_model.pkl")


def load_gesture_model(path: str = MODEL_PATH):
    """
    Returns dict with keys:
      - model
      - feature_columns
      - labels
    """
    if not os.path.exists(path):
        return None
    return joblib.load(path)