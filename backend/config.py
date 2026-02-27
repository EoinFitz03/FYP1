from dataclasses import dataclass

@dataclass
class Config:
    # Face recognition
    tolerance: float = 0.50
    downscale: float = 0.50
    model: str = "hog"
    face_every_n_frames: int = 2
    face_lost_ms: float = 800.0

    # Gesture detection
    gesture_small_width: int = 320
    gesture_smooth_window: int = 5
    gesture_min_votes: int = 2
    gesture_every_n_frames: int = 3

    # Hand tracking
    hand_lost_ms: float = 1200.0
    hand_miss_clear_count: int = 6

    # Enrolment
    enrol_min_ms_between_captures: int = 250


# Single shared instance imported everywhere
cfg = Config()