import time
import collections
from enum import Enum

import cv2
import mediapipe as mp

# Config
CAM_INDEX = 0                 # Your webcam index
MIN_DET_CONF = 0.5
MIN_TRK_CONF = 0.5
MODEL_COMPLEXITY = 0          # 0 is fastest
MAX_HANDS = 1

# Debounce / confirmation
WINDOW_SIZE = 8               # frames needed with the same gesture
COOLDOWN_SECONDS = 1.0        # time after a confirmed event before another can fire

# Gesture types
class Gesture(str, Enum):
    OPEN_PALM = "open_palm"
    FIST = "fist"
    THUMBS_UP = "thumbs_up"
    UNKNOWN = "unknown"

# Helpers
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# Landmarks index map for readability
WRIST = 0
THUMB_TIP, THUMB_IP, THUMB_MCP = 4, 3, 2
INDEX_TIP, INDEX_PIP, INDEX_MCP = 8, 6, 5
MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP = 12, 10, 9
RING_TIP, RING_PIP, RING_MCP = 16, 14, 13
PINKY_TIP, PINKY_PIP, PINKY_MCP = 20, 18, 17

FINGER_TIP_PIP = [
    (INDEX_TIP, INDEX_PIP),
    (MIDDLE_TIP, MIDDLE_PIP),
    (RING_TIP, RING_PIP),
    (PINKY_TIP, PINKY_PIP),
]

FINGER_TIP_MCP = [
    (INDEX_TIP, INDEX_MCP),
    (MIDDLE_TIP, MIDDLE_MCP),
    (RING_TIP, RING_MCP),
    (PINKY_TIP, PINKY_MCP),
]

def _dist(a, b):
    dx = a.x - b.x
    dy = a.y - b.y
    return (dx * dx + dy * dy) ** 0.5

def _is_extended_y(landmarks, tip_idx, pip_idx):
    """Return True if finger is likely extended (tip above pip in image coords)."""
    lm = landmarks
    return lm[tip_idx].y < lm[pip_idx].y

def _thumb_up_basic(landmarks):
    """
    Easier thumbs-up:
    - thumb must be clearly extended (distance-based, scale-invariant)
    - thumb tip should be above its own joints (IP + MCP), not the wrist
    - other fingers should be mostly folded (2/4 is enough)
    """
    lm = landmarks

    # A simple "hand size" reference so thresholds scale with distance to camera
    hand_size = _dist(lm[WRIST], lm[MIDDLE_MCP])
    if hand_size < 1e-6:
        return False

    # Thumb is extended enough
    thumb_len = _dist(lm[THUMB_TIP], lm[THUMB_MCP])
    thumb_extended = thumb_len > 0.45 * hand_size

    # Thumb points upward (compare to its own joints instead of wrist)
    thumb_up = (lm[THUMB_TIP].y < lm[THUMB_IP].y) and (lm[THUMB_TIP].y < lm[THUMB_MCP].y)

    # Other fingers: count folded
    # Folded if tip is below pip OR if tip is close to its MCP (distance-based)
    folded = 0
    for (tip, pip), (_, mcp) in zip(FINGER_TIP_PIP, FINGER_TIP_MCP):
        folded_by_y = lm[tip].y > lm[pip].y
        folded_by_dist = _dist(lm[tip], lm[mcp]) < 0.42 * hand_size
        if folded_by_y or folded_by_dist:
            folded += 1

    # Allow some messiness: 2+ folded fingers is enough
    folded_others = folded >= 2

    return thumb_extended and thumb_up and folded_others

def classify_gesture(hand_landmarks) -> Gesture:
    lm = hand_landmarks.landmark

    # Non-thumb fingers: count how many are extended
    extended = 0
    for tip, pip in FINGER_TIP_PIP:
        if _is_extended_y(lm, tip, pip):
            extended += 1

    # Strong, simple classes first
    if extended >= 4:
        return Gesture.OPEN_PALM

    # Check thumbs up BEFORE fist (because thumbs up often has 0 extended fingers)
    if _thumb_up_basic(lm):
        return Gesture.THUMBS_UP

    if extended == 0:
        return Gesture.FIST

    return Gesture.UNKNOWN

class DebounceState:
    """Tracks a rolling window of gestures and fires an event when stable."""
    def __init__(self, window_size=8, cooldown_seconds=1.0):
        self.window = collections.deque(maxlen=window_size)
        self.last_fire_time = 0.0
        self.cooldown_seconds = cooldown_seconds

    def update(self, g: Gesture):
        self.window.append(g)

    def stable_gesture(self):
        if len(self.window) < self.window.maxlen:
            return Gesture.UNKNOWN
        first = self.window[0]
        if first == Gesture.UNKNOWN:
            return Gesture.UNKNOWN
        if all(x == first for x in self.window):
            return first
        return Gesture.UNKNOWN

    def can_fire(self):
        return (time.time() - self.last_fire_time) >= self.cooldown_seconds

    def mark_fired(self):
        self.last_fire_time = time.time()

def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    state = DebounceState(WINDOW_SIZE, COOLDOWN_SECONDS)

    fps_t0, fps_counter = time.time(), 0
    fps_display = 0.0

    with mp_hands.Hands(
        model_complexity=MODEL_COMPLEXITY,
        max_num_hands=MAX_HANDS,
        min_detection_confidence=MIN_DET_CONF,
        min_tracking_confidence=MIN_TRK_CONF,
    ) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            current = Gesture.UNKNOWN
            if result.multi_hand_landmarks:
                hand_lms = result.multi_hand_landmarks[0]
                current = classify_gesture(hand_lms)

                mp_draw.draw_landmarks(
                    frame,
                    hand_lms,
                    mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )

            state.update(current)
            confirmed = state.stable_gesture()
            if confirmed != Gesture.UNKNOWN and state.can_fire():
                print({"event": str(confirmed), "ts": round(time.time(), 3)})
                state.mark_fired()

            fps_counter += 1
            now = time.time()
            if now - fps_t0 >= 1.0:
                fps_display = fps_counter / (now - fps_t0)
                fps_counter, fps_t0 = 0, now

            cv2.putText(frame, f"Gesture: {current}", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"FPS: {fps_display:.1f}", (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Hand Gestures", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()