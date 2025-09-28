import gunmatch
from ultralytics import YOLO
import cv2
import person_recognition
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
import os
import logging

# --- Models ---
person_model = YOLO("models/yolov8n.pt")
gun_model = YOLO("models/best3.pt")

# --- Logging ---
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Video ---
cap = cv2.VideoCapture("grocery.mp4")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  

# --- Persistent Weapon Carrier ---
json_file = "weapon_carrier_embeddings.json"
tracked_embeddings = []  # embeddings for persistent carriers
saved_embeddings = {}

# Load saved embeddings from JSON
if os.path.exists(json_file):
    try:
        with open(json_file, "r") as f:
            if os.stat(json_file).st_size > 0:
                saved_embeddings = {int(k): np.array(v) for k, v in json.load(f).items()}
            else:
                saved_embeddings = {}
    except json.JSONDecodeError:
        saved_embeddings = {}

attacker_id = None
confidence_counter = 0

def save_embeddings_to_json(embedding, weapon_carrier_id):
    global saved_embeddings
    # Store the embedding in memory as a NumPy array (optional)
    saved_embeddings[weapon_carrier_id] = embedding

    # Prepare dictionary for JSON: convert all embeddings to lists
    json_safe_dict = {k: v.tolist() if isinstance(v, np.ndarray) else v
                      for k, v in saved_embeddings.items()}

    # Save to file
    with open(json_file, "w") as f:
        json.dump(json_safe_dict, f)


# --- Thresholds ---
GUN_CONF_THRESHOLD = 0.45
PERSON_CONF_THRESHOLD = 0.4
REID_SIMILARITY_THRESHOLD = 0.6
IOU_THRESHOLD = 0.2
PROXIMITY_THRESHOLD = 150  # pixels

# --- Get frame dimensions ---
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# --- Define codec and create VideoWriter object (once, before loop) ---
out = cv2.VideoWriter(
    "output.mp4",                 # output file name
    cv2.VideoWriter_fourcc(*"mp4v"),  # codec for MP4
    30,                            # frames per second
    (frame_width, frame_height)    # frame size
)


# --- Main Loop ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    annotated_frame = frame.copy()

    # --- Detection ---
    gun_boxes = gunmatch.detect_guns(frame, gun_model, conf=GUN_CONF_THRESHOLD)
    person_boxes = gunmatch.detect_people(frame, person_model, conf=PERSON_CONF_THRESHOLD)

    # --- Match guns to people ---
    weapon_person_boxes = gunmatch.match_people_to_guns(person_boxes, gun_boxes,
                                                        iou_threshold=IOU_THRESHOLD,
                                                        proximity_threshold=PROXIMITY_THRESHOLD)

    # --- Track each person ---
    for person_box in person_boxes:
        x1, y1, x2, y2 = map(int, person_box)
        person_roi = frame[y1:y2, x1:x2]
        if person_roi.size == 0:
            continue
        person_roi_resized = cv2.resize(person_roi, (128, 256))
        embedding = person_recognition.get_embedding(person_roi_resized)

        # Check if this person matches a previously saved weapon carrier
        weapon_locked = False
        for saved_id, saved_embedding in saved_embeddings.items():
            similarity = cosine_similarity([embedding], [saved_embedding])[0][0]
            if similarity > REID_SIMILARITY_THRESHOLD:
                weapon_locked = True
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(annotated_frame, f"Weapon Locked: ID {saved_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                break

        if not weapon_locked:
            pid = person_recognition.get_person_id(embedding)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Person ID: {pid}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # --- Lock onto weapon carriers ---
    if len(gun_boxes) > 0:
        for gun_box, matched_person_box in zip(gun_boxes, weapon_person_boxes):
            # Draw gun
            gx1, gy1, gx2, gy2 = map(int, gun_box)
            cv2.rectangle(annotated_frame, (gx1, gy1), (gx2, gy2), (255, 0, 0), 2)
            cv2.putText(annotated_frame, "Weapon", (gx1, gy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # If gun is matched to a person, lock onto them
            if matched_person_box is not None:
                px1, py1, px2, py2 = map(int, matched_person_box)
                person_roi = frame[py1:py2, px1:px2]
                if person_roi.size == 0:
                    continue
                person_roi_resized = cv2.resize(person_roi, (128, 256))
                embedding = person_recognition.get_embedding(person_roi_resized)
                weapon_id = person_recognition.get_person_id(embedding)

                # Persist weapon carrier
                if weapon_id not in tracked_embeddings:
                    tracked_embeddings.append(weapon_id)
                    save_embeddings_to_json(embedding, weapon_id)

                cv2.rectangle(annotated_frame, (px1, py1), (px2, py2), (0, 0, 255), 2)
                cv2.putText(annotated_frame, f"Weapon Carrier: ID {weapon_id}", (px1, py1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)


    out.write(annotated_frame)
    cv2.imshow("Weapon Detection and Tracking", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
