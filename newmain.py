from ultralytics import YOLO
import cv2
import numpy as np
import mediapipe as mp

import person_recognition

# ADDED: import the pairing + utils from gunmatch
from gunmatch import pairs_for_frame, dedup_boxes, iou_xyxy  # ADDED

# ------------------- YOLO -------------------
model = YOLO("yolov8n.pt")

# ADDED: load your gun-only weights (adjust path/name if needed)
GUN_WEIGHTS = "best3.pt"  # ADDED
gun_model = YOLO(GUN_WEIGHTS)  # ADDED

# ADDED: optional config for pairing thresholds
CONF_GUN = 0.20       # ADDED
CONF_PERSON = 0.20    # ADDED
IMGSZ = 640           # ADDED
MATCH_IOU_THR = 0.50  # ADDED  # how close a person box must be to count as "GUN PERSON"

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

# ADDED: small label helper (doesn't change existing drawing)
def put_label(img, text, x, y, bg, fg=(255, 255, 255)):  # ADDED
    (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)  # ADDED
    cv2.rectangle(img, (x, max(0, y - th - bl - 4)), (x + tw + 6, y), bg, -1)  # ADDED
    cv2.putText(img, text, (x + 3, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, fg, 2)  # ADDED

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ADDED: BEFORE running your person loop, detect guns + persons and pair them.
    # We use your 'model' as the person detector and 'gun_model' as the gun detector.
    gun_boxes, matched_person_boxes, person_boxes_all = pairs_for_frame(  # ADDED
        frame,
        gun_model,
        model,
        conf_gun=CONF_GUN,
        conf_person=CONF_PERSON,
        imgsz=IMGSZ
    )
    # collapse duplicate matched persons (e.g., same person matched to 2 guns)  # ADDED
    unique_gun_persons = dedup_boxes(matched_person_boxes)  # ADDED

    # (Optional) draw guns so you can see detections even if persons fail  # ADDED
    for g in gun_boxes:  # ADDED
        gx1, gy1, gx2, gy2 = map(int, g)  # ADDED
        cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), (0, 0, 255), 2)  # ADDED
        put_label(frame, "GUN", gx1, gy1, bg=(0, 0, 255))  # ADDED

    # YOLO detection
    results = model(frame, device="cpu")
    annotated_frame = results[0].plot().copy()

    # ADDED: simple HUD so you know counts each frame (non-invasive)
    cv2.putText(annotated_frame,  # ADDED
                f"Guns:{len(gun_boxes)}  GunPersons:{len(unique_gun_persons)}",  # ADDED
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)  # ADDED

    # Process each detected person
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        person_roi = frame[y1:y2, x1:x2]

        # TODO: eventually call backend function here
        person_roi_resized = cv2.resize(person_roi, (128, 256))
        embedding = person_recognition.get_embedding(person_roi_resized)
        person_id = person_recognition.get_person_id(embedding)

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

            cv2.putText(
                annotated_frame,
                f"Person ID: {person_id}",
                (x1, y1 - 10),  # slightly above the top-left corner
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,  # font scale
                (0, 255, 0),  # text color (green here)
                2  # thickness
            )

        # ADDED: if THIS person box matches one of the paired "gun person" boxes, label it
        # We compare by IoU because detections from 'results' may not be identical coords as gunmatch's.
        this_box = [float(x1), float(y1), float(x2), float(y2)]  # ADDED
        is_gun_person = any(iou_xyxy(this_box, m) > MATCH_IOU_THR for m in unique_gun_persons)  # ADDED
        if is_gun_person:  # ADDED
            # draw an overlay label without removing your existing text  # ADDED
            put_label(annotated_frame, "GUN PERSON", x1, y1, bg=(0, 170, 0))  # ADDED

    # Show frame
    cv2.imshow("YOLO + MediaPipe Pose", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
