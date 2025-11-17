import cv2
import os

# Folder path for known faces
save_path = "known_faces/eoin"
os.makedirs(save_path, exist_ok=True)

# Open webcam
cap = cv2.VideoCapture(0)
print("Press SPACE to capture your photo, or Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Capture Eoin", frame)
    key = cv2.waitKey(1)

    if key == ord(" "):  # SPACE key
        file_path = os.path.join(save_path, "nathan2.jpg")
        cv2.imwrite(file_path, frame)
        print(f"Saved image: {file_path}")
        break
    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
