import cv2
from dataclasses import dataclass

from training.dataset import append_sample


@dataclass
class TrainingState:
    active: bool = False
    label: str | None = None
    target: int = 200
    count: int = 0


def try_capture_landmarks(bgr, cfg, gesture_svc, state: TrainingState):
    """
    Runs MediaPipe Hands and appends one sample to CSV if a hand is found.

    Returns:
      did_save (bool), should_send_status (bool), finished (bool)
    """
    if not state.active or not state.label:
        return False, False, False

    if gesture_svc is None or getattr(gesture_svc, "hands_detector", None) is None:
        return False, False, False

    h, w = bgr.shape[:2]
    small = bgr
    if w > cfg.gesture_small_width:
        scale = cfg.gesture_small_width / float(w)
        small = cv2.resize(bgr, (cfg.gesture_small_width, int(h * scale)))

    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    res = gesture_svc.hands_detector.process(rgb)

    if not res.multi_hand_landmarks:
        return False, False, False

    # save first hand
    hand_side = "Unknown"
    if res.multi_handedness and len(res.multi_handedness) > 0:
        hand_side = res.multi_handedness[0].classification[0].label  # "Left"/"Right"

    append_sample(state.label, hand_side, res.multi_hand_landmarks[0].landmark)
    state.count += 1

    should_send = (state.count % 25 == 0) or (state.count >= state.target)
    finished = state.count >= state.target

    if finished:
        state.active = False
        state.label = None

    return True, should_send, finished