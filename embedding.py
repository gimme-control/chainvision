from ultralytics import YOLO
import cv2
import person_recognition

# YOLO model
model = YOLO("yolo11n.pt")

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO detection
    results = model(frame, device="cuda")
    annotated_frame = results[0].plot().copy()

    # Process each detected person
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        person_roi = frame[y1:y2, x1:x2]

        # Get embedding and person ID
        person_roi_resized = cv2.resize(person_roi, (128, 256))
        embedding = person_recognition.get_embedding(person_roi_resized)
        person_id = person_recognition.get_person_id(embedding)

        # Draw person ID on frame
        cv2.putText(
            annotated_frame,
            f"Person ID: {person_id}",
            (x1 + 50, y1 - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

    # Show frame
    cv2.imshow("YOLO", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
