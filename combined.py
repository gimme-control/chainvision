import gunmatch
from ultralytics import YOLO
<<<<<<< HEAD
=======
import cv2
>>>>>>> parent of fbdc2c7 (big push incoming)
import mediapipe as mp
import person_recognition
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json  # Add this import at the top
import os
import logging
<<<<<<< HEAD
import requests

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Load YOLO models
person_model = YOLO("models/yolov8n.pt")
gun_model = YOLO("models/best3.pt")  # Use best3.pt for better gun detection
print("Gun model class names:", gun_model.names)

# Initialize webcam
cap = cv2.VideoCapture("luigi.mp4")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 300)
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()
=======

person_model = YOLO("models/yolov8n.pt")
gun_model = YOLO("models/best3.pt")

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  
>>>>>>> parent of fbdc2c7 (big push incoming)

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

# --- Persistent Weapon Carrier Tracking & ReID ---
weapon_carrier_id = None
tracked_embeddings = []  # Store embeddings only for weapon carriers
json_file = "weapon_carrier_embeddings.json"  # File to store embeddings

# Function to save embeddings to JSON and reload them
def save_embeddings_to_json(embedding, weapon_carrier_id):
    try:
        # Load existing data
        if os.path.exists(json_file):
            with open(json_file, "r") as f:
                if os.stat(json_file).st_size == 0:  # Check if file is empty
                    data = {}
                else:
                    data = json.load(f)
        else:
            data = {}
    except json.JSONDecodeError:
        data = {}  # Handle invalid JSON

    # Add new embedding
    data[str(weapon_carrier_id)] = embedding.tolist()

    # Save back to file
    with open(json_file, "w") as f:
        json.dump(data, f)

    # Reload saved embeddings into memory
    global saved_embeddings
    saved_embeddings = {int(k): np.array(v) for k, v in data.items()}

# --- Load Weapon Carrier Embeddings ---
import os

# Load saved weapon carrier embeddings from JSON
if os.path.exists(json_file):
    try:
        with open(json_file, "r") as f:
            if os.stat(json_file).st_size == 0:  # Check if file is empty
                saved_embeddings = {}
            else:
                saved_embeddings = json.load(f)
    except json.JSONDecodeError:
        saved_embeddings = {}  # Handle invalid JSON
else:
    saved_embeddings = {}

# Convert saved embeddings back to NumPy arrays
saved_embeddings = {int(k): np.array(v) for k, v in saved_embeddings.items()}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect guns and knives
    gun_boxes = gunmatch.detect_guns(frame, gun_model, conf=0.8)  # stricter threshold
    weapon_boxes = gun_boxes 

    person_boxes = gunmatch.detect_people(frame, person_model)
    weapon_person_boxes = gunmatch.match_people_to_guns(person_boxes, weapon_boxes)
    annotated_frame = frame.copy()
    person_ids = []

    # Assign IDs to all detected people
    for person_box in person_boxes:
        x1, y1, x2, y2 = map(int, person_box)
        person_roi = frame[y1:y2, x1:x2]
        if person_roi.size == 0:
            continue
        person_roi_resized = cv2.resize(person_roi, (128, 256))
        embedding = person_recognition.get_embedding(person_roi_resized)

        # Check against saved embeddings
        weapon_locked = False
        for saved_id, saved_embedding in saved_embeddings.items():
            similarity = cosine_similarity([embedding], [saved_embedding])[0][0]
            if similarity > 0.8:  # Threshold for re-identification
                weapon_locked = True
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(annotated_frame, f"Weapon Locked: ID {saved_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                break

        if not weapon_locked:
            pid = person_recognition.get_person_id(embedding)
            person_ids.append(pid)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Person ID: {pid}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Lock onto weapon carrier immediately when a weapon is detected
    for weapon_box, weapon_person_box in zip(weapon_boxes, weapon_person_boxes):
        if weapon_person_box is not None:
            x1, y1, x2, y2 = map(int, weapon_person_box)
            person_roi = frame[y1:y2, x1:x2]
            if person_roi.size == 0:
                logging.debug("Person ROI is empty. Skipping.")
                continue
            person_roi_resized = cv2.resize(person_roi, (128, 256))
            embedding = person_recognition.get_embedding(person_roi_resized)
            weapon_id = person_recognition.get_person_id(embedding)

            # Log debug information
            logging.debug(f"Weapon ID: {weapon_id}, Weapon Boxes: {weapon_boxes}, Embedding: {embedding}")

            # Validate weapon carrier assignment
            if weapon_id is not None and len(weapon_boxes) > 0:  # Ensure a valid weapon is detected
                if not any(np.array_equal(embedding, tracked) for tracked in tracked_embeddings):
                    tracked_embeddings.append(embedding)  # Save embedding only for weapon carrier
                    save_embeddings_to_json(embedding, weapon_id)  # Save to JSON
                    logging.debug(f"Weapon carrier locked: ID {weapon_id}")
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(annotated_frame, f"Weapon ID: {weapon_id}", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            else:
                logging.debug("Weapon ID is None or no valid weapon detected.")
        wx1, wy1, wx2, wy2 = map(int, weapon_box)
        cv2.rectangle(annotated_frame, (wx1, wy1), (wx2, wy2), (255, 0, 0), 2)
        cv2.putText(annotated_frame, "Weapon", (wx1, wy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imshow("Weapon Detection + ReID", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()

# --- Original Person Detection & Pose Logic (commented out) ---
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#     results = model(frame, device="cuda")
#     annotated_frame = results[0].plot().copy()
#     for box in results[0].boxes:
#         x1, y1, x2, y2 = map(int, box.xyxy[0])
#         h, w = frame.shape[:2]
#         x1, y1 = max(0, x1), max(0, y1)
#         x2, y2 = min(w, x2), min(h, y2)
#         person_roi = frame[y1:y2, x1:x2]
#         person_roi_resized = cv2.resize(person_roi, (128, 256))
#         embedding = person_recognition.get_embedding(person_roi_resized)
#         person_id = person_recognition.get_person_id(embedding)
#         rgb_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
#         pose_results = pose.process(rgb_roi)
#         if pose_results.pose_landmarks:
#             mp_drawing.draw_landmarks(
#                 person_roi,
#                 pose_results.pose_landmarks,
#                 mp_pose.POSE_CONNECTIONS
#             )
#             landmarks = pose_results.pose_landmarks.landmark
#             left_elbow_angle = angle(
#                 landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER],
#                 landmarks[mp_pose.PoseLandmark.LEFT_ELBOW],
#                 landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
#             )
#             cv2.putText(person_roi, f"LE:{int(left_elbow_angle)}", (5, 15),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
#             annotated_frame[y1:y2, x1:x2] = person_roi
#             cv2.putText(
#                 annotated_frame,
#                 f"Person ID: {person_id}",
#                 (x1, y1 - 10),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.9,
#                 (0, 255, 0),
#                 2
#             )
#     cv2.imshow("YOLO + MediaPipe Pose", annotated_frame)
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break
# cap.release()
# cv2.destroyAllWindows()
