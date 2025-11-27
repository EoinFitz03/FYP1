import time
import collections
from enum import Enum
import cv2
import mediapipe as mp
import face_recognition
from db import Database   # using SQLite DB instead of encodings.pkl

# Config
CAM_INDEX = 0                 # Your webcam index
MIN_DET_CONF = 0.5
MIN_TRK_CONF = 0.5
MODEL_COMPLEXITY = 0          # 0 is fastest
MAX_HANDS = 1

# Debounce / confirmation
WINDOW_SIZE = 8
COOLDOWN_SECONDS = 1.0

# Gesture types 
class Gesture(str, Enum):
    OPEN_PALM = "open_palm"
    THUMBS_UP = "thumbs_up"
    UNKNOWN = "unknown"

# MediaPipe helpers
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# Landmark indexes
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

    # OPEN PALM
    if extended >= 4:
        return Gesture.OPEN_PALM

    # THUMBS UP
    if extended == 0 and _thumb_up_basic(lm):
        return Gesture.THUMBS_UP

    return Gesture.UNKNOWN


class DebounceState:
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


# Load encodings from DB
db = Database("system.db")
known_encodings, known_names = db.load_all_encodings()
print(f"Loaded {len(known_encodings)} encodings from DB.")

# Main loop
def main():
    cap = cv2.VideoCapture(CAM_INDEX)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

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

            # 1) HAND GESTURES
            rgb_full = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb_full)

            current_gesture = Gesture.UNKNOWN
            if result.multi_hand_landmarks:
                hand_lms = result.multi_hand_landmarks[0]
                current_gesture = classify_gesture(hand_lms)

                mp_draw.draw_landmarks(
                    frame,
                    hand_lms,
                    mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )

            state.update(current_gesture)
            confirmed = state.stable_gesture()

            # 2) FACE RECOGNITION
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(rgb_small)
            face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

            current_names = []

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(known_encodings, face_encoding)
                name = "Unknown"

                if True in matches:
                    match_index = matches.index(True)
                    name = known_names[match_index]

                current_names.append(name)

                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            primary_name = current_names[0] if len(current_names) > 0 else "None"

            # 3) Combined gesture + face event
            if confirmed != Gesture.UNKNOWN and state.can_fire():
                event_info = {
                    "gesture": confirmed.value,
                    "user": primary_name,
                    "ts": round(time.time(), 3),
                }
                print(event_info)

                try:
                    db.add_event(
                        user_name=primary_name,
                        gesture=confirmed.value,
                        action="gesture_event"
                    )
                except:
                    pass

                state.mark_fired()

            # 4) HUD / FPS
            fps_counter += 1
            now = time.time()
            if now - fps_t0 >= 1.0:
                fps_display = fps_counter / (now - fps_t0)
                fps_counter, fps_t0 = 0, now

            cv2.putText(frame, f"Gesture: {current_gesture.value}", (12, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.putText(frame, f"User: {primary_name}", (12, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

            cv2.putText(frame, f"FPS: {fps_display:.1f}", (12, 88),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Face + Hand Gestures", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    db.close()


if __name__ == "__main__":
    main()
