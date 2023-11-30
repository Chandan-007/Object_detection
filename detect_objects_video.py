import cv2
import numpy as np
import time

print("Loading model...")

# Load pre-trained SSD model with its configuration and weights
net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt', 'MobileNetSSD_deploy.caffemodel')
print("Model loaded successfully.")

# Load the class labels
with open('MobileNetSSD_classes.txt', 'r') as f:
    classes = f.read().strip().split('\n')

# Open a video capture object
cap = cv2.VideoCapture('videos/17.mp4')

# Set the display window dimensions
cv2.namedWindow("Object Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Object Detection", 1200, 780)

# Variables for tracking
previous_time = time.time()

# Dictionary to store object positions and colors
object_data = {}

def get_random_color():
    return np.random.randint(0, 255, size=3).tolist()

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()

    # Break the loop if the video has ended
    if not ret:
        break

    # Resize the frame to a fixed width and height
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)

    # Set the input to the neural network
    net.setInput(blob)

    # Perform object detection and get the bounding boxes and confidence levels
    detections = net.forward()

    # Loop over the detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections by confidence level
        if confidence > 0.3:
            # Get the class index
            class_index = int(detections[0, 0, i, 1])

            # Get the class name and bounding box coordinates
            class_name = classes[class_index]
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            (startX, startY, endX, endY) = box.astype("int")

            # Draw the bounding box, class name, and confidence on the frame
            label = f"{class_name}: {confidence * 100:.2f}%"

            # Assign a random color to the class (if not assigned yet)
            if class_name not in object_data:
                object_data[class_name] = {'color': get_random_color(), 'position': (startX, startY)}
            else:
                # Calculate distance between current and previous positions
                distance = np.sqrt((startX - object_data[class_name]['position'][0]) ** 2 +
                                   (startY - object_data[class_name]['position'][1]) ** 2)

                # Calculate speed in pixels per second
                speed = distance / (time.time() - previous_time)
                object_data[class_name]['position'] = (startX, startY)

                # Display the speed on the frame
                cv2.putText(frame, f"Speed ({class_name}): {speed:.2f} pixels/second", (startX, endY + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, object_data[class_name]['color'], 2)

            color = object_data[class_name]['color']
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the resulting frame
    cv2.imshow("Object Detection", frame)

    # Update tracking variables
    previous_time = time.time()

    # Handle keyboard input
    key = cv2.waitKey(1) & 0xFF

    # Stop the video when 'q' key is pressed or window is closed
    if key == ord('q') or cv2.getWindowProperty("Object Detection", cv2.WND_PROP_VISIBLE) < 1:
        break

# Release the video capture object and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
