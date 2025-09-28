import gunmatch
from ultralytics import YOLO
import cv2
# import image_utils
import person_recognition
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
import os
import logging


picture_counter = 0 
# --- Models ---
# Assuming these models are accessible in your environment
person_model = YOLO("models/yolov8n.pt")
gun_model = YOLO("models/best.pt")

# --- Logging ---
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Video ---
cap = cv2.VideoCapture("warehouse.mp4")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600)  
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 350)  

# --- Persistent Weapon Carrier ---
json_file = "weapon_carrier_embeddings.json"
tracked_embeddings = []  # IDs for persistent carriers
saved_embeddings = {}

# Load saved embeddings from JSON
if os.path.exists(json_file):
    try:
        with open(json_file, "r") as f:
            if os.stat(json_file).st_size > 0:
                # Load JSON and convert list back to NumPy array
                saved_embeddings = {int(k): np.array(v) for k, v in json.load(f).items()}
            else:
                saved_embeddings = {}
    except json.JSONDecodeError:
        logging.warning(f"Could not decode JSON from {json_file}. Starting with empty embeddings.")
        saved_embeddings = {}

attacker_id = None
confidence_counter = 0

def save_embeddings_to_json(embedding, weapon_carrier_id):
    """Saves a new weapon carrier's embedding to the persistent JSON file."""
    global saved_embeddings
    # Store the embedding in memory
    saved_embeddings[weapon_carrier_id] = embedding

    # Prepare dictionary for JSON: convert all embeddings to lists
    json_safe_dict = {str(k): v.tolist() if isinstance(v, np.ndarray) else v
                      for k, v in saved_embeddings.items()}

    # Save to file
    with open(json_file, "w") as f:
        json.dump(json_safe_dict, f, indent=4)


# --- Thresholds ---
GUN_CONF_THRESHOLD = 0.25
PERSON_CONF_THRESHOLD = 0.6
REID_SIMILARITY_THRESHOLD = 0.70
IOU_THRESHOLD = 0.4
PROXIMITY_THRESHOLD = 150  # pixels

# --- Get frame dimensions ---
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# --- Define codec and create VideoWriter object (Crucial Fix Applied Here) ---
# FIX: Switched to XVID codec and .avi extension for robust writing across platforms.
out = cv2.VideoWriter(
    "output_annotated.avi",         # Changed to .avi
    cv2.VideoWriter_fourcc(*"XVID"),  # Changed codec to XVID
    30,                            # frames per second
    (frame_width, frame_height)    # frame size
)

# Check if VideoWriter was initialized successfully
if not out.isOpened():
    logging.error("VideoWriter failed to open. Try a different codec (e.g., 'MJPG', 'DIVX', 'mp4v') or ensure FFmpeg/codec packages are installed.")
    cap.release()
    cv2.destroyAllWindows()
    # Exit gracefully if we can't write the video
    exit()

logging.info(f"VideoWriter initialized for output_annotated.avi with dimensions: {frame_width}x{frame_height}")

# --- Main Loop ---
while True:
    ret, frame = cap.read()
    if not ret:
        logging.info("End of video stream.")
        break

    annotated_frame = frame.copy()

    # --- Detection ---
    gun_boxes = gunmatch.detect_guns(frame, gun_model, conf=GUN_CONF_THRESHOLD)
    person_boxes = gunmatch.detect_people(frame, person_model, conf=PERSON_CONF_THRESHOLD)

    # --- Match guns to people ---
    weapon_person_boxes = gunmatch.match_people_to_guns(person_boxes, gun_boxes,
                                                        iou_threshold=IOU_THRESHOLD,
                                                        proximity_threshold=PROXIMITY_THRESHOLD)

    # --- Track each person and check against persistent carriers ---
    for person_box in person_boxes:
        x1, y1, x2, y2 = map(int, person_box)
        person_roi = frame[y1:y2, x1:x2]
        if person_roi.size == 0:
            continue
        
        # Ensure ROI is valid for person_recognition.get_embedding (128x256)
        if person_roi.shape[0] < 2 or person_roi.shape[1] < 2:
            continue
            
        person_roi_resized = cv2.resize(person_roi, (128, 256))
        embedding = person_recognition.get_embedding(person_roi_resized)

        # Check if this person matches a previously saved weapon carrier
        weapon_locked = False
        current_person_id = None
        
        # Check against saved embeddings
        for saved_id, saved_embedding in saved_embeddings.items():
            similarity = cosine_similarity([embedding], [saved_embedding])[0][0]
            if similarity > REID_SIMILARITY_THRESHOLD:
                weapon_locked = True
                current_person_id = saved_id
                
                # Draw locked status (RED)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(annotated_frame, f"Locked: ID 4", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                # Take pictures if we haven't done it yet 5 times:
                if picture_counter < 5:
                    image_utils.save_clipped_person(frame, (x1, y1, x2, y2), saved_id, picture_counter)
                    picture_counter += 1
                break
    
        # If not locked, treat as a new or tracked person (GREEN)
        if not weapon_locked:
            # Use ReID to get a temporary ID for tracking
            current_person_id = person_recognition.get_person_id(embedding) 
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Person ID: {current_person_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


    # --- Lock onto weapon carriers (Primary Detection) ---
    for gun_box, matched_person_box in zip(gun_boxes, weapon_person_boxes):
        # Draw gun (BLUE)
        gx1, gy1, gx2, gy2 = map(int, gun_box)
        cv2.rectangle(annotated_frame, (gx1, gy1), (gx2, gy2), (255, 0, 0), 2)
        cv2.putText(annotated_frame, "Weapon", (gx1, gy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # If gun is matched to a person, confirm them as a carrier
        if matched_person_box is not None:
            px1, py1, px2, py2 = map(int, matched_person_box)
            person_roi = frame[py1:py2, px1:px2]
            if person_roi.size == 0:
                continue
            
            # Ensure ROI is valid
            if person_roi.shape[0] < 2 or person_roi.shape[1] < 2:
                continue

            person_roi_resized = cv2.resize(person_roi, (128, 256))
            embedding = person_recognition.get_embedding(person_roi_resized)
            weapon_id = person_recognition.get_person_id(embedding)

            # Persist weapon carrier if new
            if weapon_id not in tracked_embeddings:
                tracked_embeddings.append(weapon_id)
                save_embeddings_to_json(embedding, weapon_id)
        

            # Re-draw the carrier box in the primary carrier color (BRIGHT RED)
            cv2.rectangle(annotated_frame, (px1, py1), (px2, py2), (0, 0, 255), 4) # Thicker line
            cv2.putText(annotated_frame, f"WEAPON CARRIER: ID 4", (px1, py1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)


    out.write(annotated_frame)
    cv2.imshow("Weapon Detection and Tracking", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# --- Cleanup ---
out.release()
cv2.destroyAllWindows()
logging.info("Video processing finished and resources released.")
image_utils.generate_summary(image_utils.image_list)
