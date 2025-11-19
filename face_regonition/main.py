import cv2

def main():
    # Trying the Lenovo webcam
    cap = cv2.VideoCapture(0)  # this allows you to pick your camera 

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Lenovo webcam opened successfully! Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Retrying...")
             #will give this error message f teh camera is not detected 

        # Show the feed
        cv2.imshow("Smart Doorbell - Lenovo Webcam", frame)

        # Quit with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break    # this allows you to press q and teh windwow will disappear 

    cap.release()
    cv2.destroyAllWindows()  # calling functions 

if __name__ == "__main__":
    main()
