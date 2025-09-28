from ultralytics import YOLO
import cv2
import numpy as np
import mediapipe as mp
import image_utils

import person_recognition

# ------------------- YOLO -------------------
model = YOLO("models/yolov8n.pt")

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# ------------------- MediaPipe Pose -------------------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Function to compute angles
def angle(a, b, c):
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba)*np.linalg.norm(bc)+1e-6)
    return np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))

frame_count = 5

while frame_count > 0:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO detection
    results = model(frame, device="cpu")
    annotated_frame = results[0].plot().copy()

    # Process each detected person
    for box in results[0].boxes:
        cls = int(box.cls[0])  # class index
        conf = float(box.conf[0])  # confidence score

        # Skip anything that's not a person or too low confidence
        if cls != 0 or conf < 0.75:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        person_roi = frame[y1:y2, x1:x2]

        image_utils.save_clipped_person(frame, (x1, y1, x2, y2), 0, frame_count)
        frame_count -= 1
        break

cap.release()
cv2.destroyAllWindows()