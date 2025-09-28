import cv2
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("models/best3.pt")
class_list = model.names  # Get class names from the model
print("Loaded class names:", class_list)

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame")
        break

    # Run YOLO detection
    results = model(frame, conf=0.5)
    if results[0].boxes is None:
        cv2.imshow('Gun Detection', frame)
        if cv2.waitKey(1) & 0xFF == 27:  # Press 'ESC' to exit
            break
        continue

    # Process detections
    boxes = results[0].boxes.data
    class_ids = results[0].boxes.cls

    for box, class_id in zip(boxes, class_ids):
        x1, y1, x2, y2 = map(int, box[:4].tolist())
        label = class_list[int(class_id)]

        # Draw bounding box and label if a gun is detected
        if label == "guns":
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                frame,
                "Gun Detected",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

    # Display the frame
    cv2.imshow('Gun Detection', frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Press 'ESC' to exit
        break

# Release resources
cap.release()
cv2.destroyAllWindows()