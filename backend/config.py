from dataclasses import dataclass
# Import dataclass so configuration values can be grouped into one simple settings object
@dataclass
class Config: # Central backend configuration for recognition, tracking, and enrolment behaviour
    # Face recognition
    tolerance: float = 0.50 # Central backend configuration for recognition, tracking, and enrolment behaviour
    downscale: float = 0.50 # Resize factor used to shrink frames before face detection for faster processing
    model: str = "hog" # Face-detection model used by the face_recognition library
    face_every_n_frames: int = 2 # Run face recognition once every N frames instead of on every frame
    face_lost_ms: float = 800.0 # Keep the previous recognised face for this many milliseconds before resetting to Unknown

    # Gesture detection
    gesture_small_width: int = 320 # Target width used when shrinking frames before gesture processing
    gesture_smooth_window: int = 5 # Number of recent gesture predictions stored for majority-vote smoothing
    gesture_min_votes: int = 2 # Minimum number of votes needed before a gesture is accepted from the smoothing history
    gesture_every_n_frames: int = 3 # Run gesture recognition once every N frames to reduce processing cost

    # Hand tracking
    hand_lost_ms: float = 1200.0 # Time limit before a stored gesture expires after the hand is no longer seen
    hand_miss_clear_count: int = 6 # Number of consecutive missed gesture detections allowed before clearing gesture state

    # Enrolment
    enrol_min_ms_between_captures: int = 250 # Minimum time gap between saved face samples during enrolment


# Single shared instance imported everywhere
cfg = Config()