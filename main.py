from ultralytics import YOLO
import cv2
import numpy as np
import mediapipe as mp

# ------------------- YOLO -------------------
model = YOLO("yolov8n.pt").to("cuda")

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
        person_roi = frame[y1:y2, x1:x2]

        # Convert to RGB for MediaPipe
        rgb_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(rgb_roi)

        if pose_results.pose_landmarks:
            # Draw landmarks
            mp_drawing.draw_landmarks(
                person_roi,
                pose_results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

            # Example: left elbow angle
            landmarks = pose_results.pose_landmarks.landmark
            left_elbow_angle = angle(
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER],
                landmarks[mp_pose.PoseLandmark.LEFT_ELBOW],
                landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
            )
            cv2.putText(person_roi, f"LE:{int(left_elbow_angle)}", (5, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Replace the annotated ROI in the frame
            annotated_frame[y1:y2, x1:x2] = person_roi

    # Show frame
    cv2.imshow("YOLO + MediaPipe Pose", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
