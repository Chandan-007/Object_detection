import cv2
import face_recognition

# Open a video capture object
cap = cv2.VideoCapture('videos/6.mp4')

# Set the display window dimensions
cv2.namedWindow("Face and Gender Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Face and Gender Detection", 1200, 780)

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()

    # Break the loop if the video has ended
    if not ret:
        break

    # Find all face locations in the frame
    face_locations = face_recognition.face_locations(frame)

    # Loop over the detected faces
    for face_location in face_locations:
        # Extract the face region
        top, right, bottom, left = face_location
        face = frame[top:bottom, left:right]

        # Use face_recognition to estimate age and gender
        face_encoding = face_recognition.face_encodings(frame, [face_location])[0]
        face_data = face_recognition.face_landmarks(frame, [face_location])[0]
        age = "Unknown"  # Face recognition library does not directly provide age
        gender = "Unknown"  # Face recognition library does not directly provide gender

        # Display the bounding box, age, and gender on the frame
        label = f"Age: {age}, Gender: {gender}"
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow("Face and Gender Detection", frame)

    # Handle keyboard input
    key = cv2.waitKey(1) & 0xFF

    # Stop the video when 'q' key is pressed or window is closed
    if key == ord('q') or cv2.getWindowProperty("Face and Gender Detection", cv2.WND_PROP_VISIBLE) < 1:
        break

# Release the video capture object and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
