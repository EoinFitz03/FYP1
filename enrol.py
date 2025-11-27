import cv2
import face_recognition
import numpy as np
from db import Database


def enrol_person(name, num_samples=10):
    db = Database("system.db")

    # Check if user already exists
    user_id = db.get_user_id(name)
    if user_id is None:
        user_id = db.add_user(name)
        print(f"Created new user '{name}' with id {user_id}")
    else:
        print(f"Adding more encodings for existing user '{name}' (id {user_id})")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    collected = []
    print("Look at the camera. Press 'q' to cancel.")

    while len(collected) < num_samples:
        ret, frame = cap.read()
        if not ret:
            break

        # Show the live feed
        cv2.imshow("Enrolment - Look at the camera", frame)

        # Downscale for speed
        small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small)
        face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

        # Take the first face found in the frame (if any)
        if face_encodings:
            enc = face_encodings[0]
            collected.append(enc)
            print(f"Captured {len(collected)}/{num_samples} samples")

        # Quit with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Enrolment cancelled.")
            cap.release()
            cv2.destroyAllWindows()
            db.close()
            return

    cap.release()
    cv2.destroyAllWindows()

    if not collected:
        print("No face encodings collected. Try again.")
        db.close()
        return

    # Average the collected encodings to get a stable one
    mean_encoding = np.mean(collected, axis=0)
    db.add_face_encoding(user_id, mean_encoding)
    db.close()

    print(f"Saved face encoding for '{name}'.")


if __name__ == "__main__":
    person_name = input("Enter the name of the person to enrol: ").strip()
    if person_name:
        enrol_person(person_name)
    else:
        print("No name entered. Exiting.")
