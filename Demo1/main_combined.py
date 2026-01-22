import time # for FPS and cooldown timing
import collections # for thr deque 
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
WINDOW_SIZE = 8 # 8 frames before it trust the hand gesture 
COOLDOWN_SECONDS = 1.0 # how long before firing another event 

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
] # describes the point in the hand joints
# 21 landmarks from media pipe hands 

def _is_extended_y(landmarks, tip_idx, pip_idx):
    # In MediaPipe image coordinates:
    # - y=0 is the TOP of the image
    # - y gets bigger as you go DOWN
    #
    # If the fingertip is ABOVE the PIP joint (tip y < pip y),
    # we treat that finger as "extended" (pointing up / open).
    lm = landmarks
    return lm[tip_idx].y < lm[pip_idx].y

def _thumb_up_basic(landmarks):
    lm = landmarks
    up = lm[THUMB_TIP].y < lm[WRIST].y

    folded_others = True
    for tip, pip in FINGER_TIP_PIP:
        if lm[tip].y < lm[pip].y:
            folded_others = False #everything else folded like a fist 
            break

    return up and folded_others
# checks for a thumb joint pointing up

def classify_gesture(hand_landmarks) -> Gesture:
    lm = hand_landmarks.landmark

    # Count non-thumb fingers extended, check if gesture is a landmark
    extended = 0
    for tip, pip in FINGER_TIP_PIP:
        if _is_extended_y(lm, tip, pip):
            extended += 1

    # OPEN PALM
    if extended >= 4:
        return Gesture.OPEN_PALM # if 4 or 4 fingers are extended it is an open palm 

    # THUMBS UP
    if extended == 0 and _thumb_up_basic(lm): # no finger extended it is a thimb up 
        return Gesture.THUMBS_UP

    return Gesture.UNKNOWN


class DebounceState:
    def __init__(self, window_size=8, cooldown_seconds=1.0):
        self.window = collections.deque(maxlen=window_size) #last 8 frames 
        self.last_fire_time = 0.0 # when last event was logged 
        self.cooldown_seconds = cooldown_seconds # how long to wait for another event 

    def update(self, g: Gesture):
        self.window.append(g) # checks if the gestire is the saem 

    def stable_gesture(self):
        if len(self.window) < self.window.maxlen:
            return Gesture.UNKNOWN # if there is not enough trusted frames it is unknown
        first = self.window[0] # no flciker between gestures, this only works if the gesture is there 
        # for multiple frames not just 1 
        if first == Gesture.UNKNOWN:
            return Gesture.UNKNOWN
        if all(x == first for x in self.window):
            return first
        # are all items in the window the same ?
        return Gesture.UNKNOWN 
        # if anyhting fails in the 8 frames gesture is unknown

    def can_fire(self):
        return (time.time() - self.last_fire_time) >= self.cooldown_seconds
    # time in seconds, when an event was last logged, greater than 1 second it can log again 

    def mark_fired(self):
        self.last_fire_time = time.time()
        # event logged, start cooldown 


# Load encodings from DB
db = Database("system.db")
known_encodings, known_names = db.load_all_encodings()
print(f"Loaded {len(known_encodings)} encodings from DB.")
# loading the known face encodings and names from the DB 

# Main loop
def main():
    cap = cv2.VideoCapture(CAM_INDEX)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    # opens camera 
    state = DebounceState(WINDOW_SIZE, COOLDOWN_SECONDS)
    # makes a debounce state for the gestures 
    fps_t0, fps_counter = time.time(), 0
    fps_display = 0.0
    # calculates 
    with mp_hands.Hands(
        model_complexity=MODEL_COMPLEXITY,
        max_num_hands=MAX_HANDS,
        min_detection_confidence=MIN_DET_CONF, # how confident the model is that it is a hand 
        min_tracking_confidence=MIN_TRK_CONF, # confidence on tracking teh hadn frame by frame 
    ) as hands:

        while True:
            ok, frame = cap.read()
            if not ok:
                break # breaks if something fails 

            frame = cv2.flip(frame, 1) # this flips the camera to make it more real 

            # 1) HAND GESTURES
            rgb_full = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
            # converts the frame into what media pipe expects which is RGB but open cv gives BGR
            result = hands.process(rgb_full)
            # this is where the ai runs, it processes the frame to get a landmark

            current_gesture = Gesture.UNKNOWN
            # starts with unknown every frame 

            if result.multi_hand_landmarks:
                hand_lms = result.multi_hand_landmarks[0]
                current_gesture = classify_gesture(hand_lms)
                # takes th efirst hand it sees, then classifies the gesture into the 3 

                mp_draw.draw_landmarks(
                    frame,
                    hand_lms,
                    mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )
                # this does not affect anything only a visual of the joints 

            state.update(current_gesture)
            confirmed = state.stable_gesture()
            # this confirms the gesture and updates it very important !

            # 2) FACE RECOGNITION
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            # fewer pixels to process at 25 so leads to faster detection 
            rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            # same as hands 
            face_locations = face_recognition.face_locations(rgb_small)
            # detects wheer the face is and out puts bounding boxes for right or left
            # gets teh rectangle 
            face_encodings = face_recognition.face_encodings(rgb_small, face_locations)
            # creates the 128 encodings fo rthe face location


            current_names = []

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                # zip paiirs up the location ans the encodings 
                matches = face_recognition.compare_faces(known_encodings, face_encoding)
                name = "Unknown"
                # compares the encodings to teh DB to see if there is a match 

                if True in matches:
                    match_index = matches.index(True)
                    name = known_names[match_index]
                # if there is a match find it and grab the name 
                current_names.append(name)

                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                # draw a box around the face 

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            primary_name = current_names[0] if len(current_names) > 0 else "None"
            # this simply makes the box green and puts the perosns namae above it 

            # 3) Combined gesture + face event
            if confirmed != Gesture.UNKNOWN and state.can_fire():
                # this only run sif gesture is real this avoids spamming events 

                event_info = {
                    "gesture": confirmed.value,
                    "user": primary_name,
                    "ts": round(time.time(), 3),
                }
                print(event_info)
                # info for debugging comes up when you get a log in the terminal 

                try:
                    db.add_event(
                        user_name=primary_name,
                        gesture=confirmed.value,
                        action="gesture_event"
                    )
                    # simply just saves these events to the database 
                except:
                    pass
                    # this is a precaution to stop everything from crashing if somehting goes wrong 
                state.mark_fired()
                # starts cooldown timer 

            # 4) HUD / FPS
            fps_counter += 1 
            # every loop counts the frame 
            now = time.time()
            if now - fps_t0 >= 1.0:
                fps_display = fps_counter / (now - fps_t0)
                fps_counter, fps_t0 = 0, now
                # update sthe FPS once per second 
                # every 1 second it clavulates the frames then resets the counter 

            cv2.putText(frame, f"Gesture: {current_gesture.value}", (12, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                # show gesture ur currently seeing 

            cv2.putText(frame, f"User: {primary_name}", (12, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
                # show the user you currently seeing 
            cv2.putText(frame, f"FPS: {fps_display:.1f}", (12, 88),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            # show FPS 
            cv2.imshow("Face + Hand Gestures", frame)
            # display the latest on the window

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    db.close()


if __name__ == "__main__":
    main()
