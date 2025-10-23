import cv2
import face_recognition

# Open webcam 
video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("Could not open webcam.")
    exit()

print("Webcam opened successfully. Press 'q' to quit")

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Frame not captured, check camera connection.")
        break

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert from BGR (OpenCV) to RGB (face_recognition)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Detect faces
    face_locations = face_recognition.face_locations(rgb_small_frame)

    # Draw rectangles around detected faces
    for (top, right, bottom, left) in face_locations:
        # Scale back up since we resized
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

    # Display the result
    cv2.imshow('Smart Doorbell - Live Feed', frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
