import face_recognition
import os
import pickle

KNOWN_FACES_DIR = "known_faces"
ENCODINGS_FILE = "encodings.pkl"
known_encodings = []
known_names = []

print("Loading known faces...")

for person_name in os.listdir(KNOWN_FACES_DIR):
    person_folder = os.path.join(KNOWN_FACES_DIR, person_name)

    if not os.path.isdir(person_folder):
        continue

    print(f"Processing {person_name}...")

    for filename in os.listdir(person_folder):
        filepath = os.path.join(person_folder, filename)
        image = face_recognition.load_image_file(filepath)

        encodings = face_recognition.face_encodings(image)
        if len(encodings) > 0:
            known_encodings.append(encodings[0])
            known_names.append(person_name)
            print(f"Added {filename}")
        else:
            print(f"No face found in {filename}, skipping...")

with open(ENCODINGS_FILE, "wb") as f:
    pickle.dump((known_encodings, known_names), f)

print("Training complete. Encodings saved to encodings.pkl")
