import time
import collections
from enum import Enum

import cv2
import mediapipe as mp
import face_recognition
import pickle

# =========================
# Config
# =========================
CAM_INDEX = 0                 # Your webcam index
MIN_DET_CONF = 0.5
MIN_TRK_CONF = 0.5
MODEL_COMPLEXITY = 0          # 0 is fastest
MAX_HANDS = 1

# Debounce / confirmation (for gestures)
WINDOW_SIZE = 8               # frames needed with the same gesture
COOLDOWN_SECONDS = 1.0        # time after a confirmed event before another can fire

# =========================
# Gesture types
# =========================
class Gesture(str, Enum):
    OPEN_PALM = "open_palm"
    FIST = "fist"
    THUMBS_UP = "thumbs_up"
    UNKNOWN = "unknown"

# =========================
# MediaPipe helpers
# =========================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# Landmarks index map for readability
WRIST = 0
THUMB_TIP, THUMB_IP, THUMB_MCP = 4, 3, 2
INDEX_TIP, INDEX_PIP = 8, 6
MIDDLE_TIP, MIDDLE_PIP = 12, 10
RING_TIP, RING_PIP = 16, 14
PINKY_TIP, PINKY_PIP = 20, 18

FINGER_TIP_PIP = [
    (INDEX_TIP, INDEX_PIP),
    (MIDDLE_TIP, MIDDLE_PIP),
    (RING_TIP, RING_PIP),
    (PINKY_TIP, PINKY_PIP),
]

def _is_extended_y(landmarks, tip_idx, pip_idx):
    lm = landmarks
    return lm[tip_idx].y < lm[pip_idx].y

def _thumb_up_basic(landmarks):
    lm = landmarks
    up = lm[THUMB_TIP].y < lm[WRIST].y

    folded_others = True
    for tip, pip in FINGER_TIP_PIP:
        if lm[tip].y < lm[pip].y:
            folded_others = False
            break

    return up and folded_others

def classify_gesture(hand_landmarks) -> Gesture:

    lm = hand_landmarks.landmark

    # Count non-thumb fingers extended
    extended = 0
    for tip, pip in FINGER_TIP_PIP:
        if _is_extended_y(lm, tip, pip):
            extended += 1

    # Strong, simple classes first
    if extended >= 4:
        return Gesture.OPEN_PALM
    if extended == 0:
        # maybe thumbs up? If not, it's a fist.
        return Gesture.THUMBS_UP if _thumb_up_basic(lm) else Gesture.FIST

    # Otherwise unknown for now
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
        """Return a stable gesture if the last N frames all match and are not UNKNOWN."""
        if len(self.window) < self.window.maxlen:
            return Gesture.UNKNOWN
        first = self.window[0]
        if first == Gesture.UNKNOWN:
            return Gesture.UNKNOWN
        if all(x == first for x in self.window):
            return first
        return Gesture.UNKNOWN

    def can_fire(self):
        """Check if enough time has passed since last confirmed event."""
        return (time.time() - self.last_fire_time) >= self.cooldown_seconds

    def mark_fired(self):
        self.last_fire_time = time.time()

# =========================
# Load face encodings
# =========================
with open("encodings.pkl", "rb") as f:
    known_encodings, known_names = pickle.load(f)

print("Encodings loaded successfully.")

# =========================
# Main loop (combined)
# =========================
def main():
    cap = cv2.VideoCapture(CAM_INDEX)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    state = DebounceState(WINDOW_SIZE, COOLDOWN_SECONDS)

    fps_t0, fps_counter = time.time(), 0
    fps_display = 0.0

    # Mediapipe Hands context manager
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

            # Mirror for a selfie view (IMPORTANT: both face + hands use this same frame)
            frame = cv2.flip(frame, 1)

            # -------------------------
            # 1) HAND GESTURES
            # -------------------------
            rgb_full = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb_full)

            current_gesture = Gesture.UNKNOWN
            if result.multi_hand_landmarks:
                # Only use the first detected hand for now
                hand_lms = result.multi_hand_landmarks[0]
                current_gesture = classify_gesture(hand_lms)

                # Draw landmarks for feedback
                mp_draw.draw_landmarks(
                    frame,
                    hand_lms,
                    mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )

            # Debounce / confirmation of gesture
            state.update(current_gesture)
            confirmed = state.stable_gesture()

            # -------------------------
            # 2) FACE RECOGNITION
            # -------------------------
            # Use a smaller frame for speed (still from the same flipped frame)
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(rgb_small)
            face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

            # Track names for HUD (e.g. first recognized face)
            current_names = []

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(known_encodings, face_encoding)
                name = "Unknown"

                if True in matches:
                    match_index = matches.index(True)
                    name = known_names[match_index]

                current_names.append(name)

                # Scale back up since we used a 1/4 size frame for detection
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Draw face box + name on the main frame
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Get a single "primary" name for HUD / events
            primary_name = current_names[0] if len(current_names) > 0 else "None"

            # -------------------------
            # 3) Fire gesture event (with face info)
            # -------------------------
            if confirmed != Gesture.UNKNOWN and state.can_fire():
                # This is where you trigger an action that uses BOTH:
                # - confirmed gesture
                # - currently recognized face (primary_name)
                print({
                    "event": str(confirmed),
                    "user": primary_name,
                    "ts": round(time.time(), 3),
                })
                state.mark_fired()

            # -------------------------
            # 4) HUD / FPS / Labels
            # -------------------------
            fps_counter += 1
            now = time.time()
            if now - fps_t0 >= 1.0:
                fps_display = fps_counter / (now - fps_t0)
                fps_counter, fps_t0 = 0, now

            # Gesture HUD
            cv2.putText(frame, f"Gesture: {current_gesture.value}", (12, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Show primary recognized name (if any)
            cv2.putText(frame, f"User: {primary_name}", (12, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

            # FPS HUD
            cv2.putText(frame, f"FPS: {fps_display:.1f}", (12, 88),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Face + Hand Gestures", frame)

            # Quit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
