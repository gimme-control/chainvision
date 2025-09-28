import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
import person_recognition
from sklearn.metrics.pairwise import cosine_similarity
import json
import os
import logging
import requests

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Load YOLO models
person_model = YOLO("models/yolov8n.pt")
gun_model = YOLO("models/best3.pt")
print("Gun model class names:", gun_model.names)

# Initialize webcam
cap = cv2.VideoCapture("luigi.mp4")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 300)
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

# Function to compute angles
def angle(a, b, c):
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))

# --- NEW: Function to calculate IoU between two bounding boxes (xyxy format) ---
def iou_xyxy(box1, box2):
    """Calculate the Intersection over Union (IoU) of two bounding boxes in xyxy format."""
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0
    return inter_area / union_area

# --- Integrated People Detection (previously gunmatch.detect_people) ---
def detect_people(frame, person_model, conf_threshold=0.2):
    """Detects people in the frame using the YOLO model."""
    results = person_model(frame, conf=conf_threshold, classes=[0])  # class 0 is 'person'
    person_data = [] # Stores (box, embedding, pid)
    if results[0].boxes is not None:
        for box in results[0].boxes.xyxy:
            person_box = box.cpu().numpy()
            x1, y1, x2, y2 = map(int, person_box)
            person_roi = frame[y1:y2, x1:x2]
            
            if person_roi.size == 0 or x2 <= x1 or y2 <= y1:
                continue

            person_roi_resized = cv2.resize(person_roi, (128, 256))
            embedding = person_recognition.get_embedding(person_roi_resized)
            
            # Get general Person ID (from person_recognition internal DB)
            pid = person_recognition.get_person_id(embedding)
            
            person_data.append({
                'box': person_box,
                'embedding': embedding,
                'pid': pid,
                'is_weapon_carrier': False,
                'reid_locked_id': None
            })
    return person_data

# --- Integrated Gun-to-Person Matching (previously gunmatch.match_people_to_guns) ---
def match_people_to_guns(person_data, gun_boxes_xyxy, iou_threshold=0.01):
    """Matches a gun box to the person box with the highest overlap (IoU)."""
    # Create a list to store the matched person object for each gun
    weapon_person_matches = [None] * len(gun_boxes_xyxy)
    
    for i, gun_box in enumerate(gun_boxes_xyxy):
        max_iou = iou_threshold
        best_match_person_data = None
        
        for person in person_data:
            person_box = person['box']
            current_iou = iou_xyxy(gun_box, person_box)
            
            # Check proximity (optional but good for filtering false matches)
            wx, wy = (gun_box[0] + gun_box[2]) / 2, (gun_box[1] + gun_box[3]) / 2
            px, py = (person_box[0] + person_box[2]) / 2, (person_box[1] + person_box[3]) / 2
            distance = np.sqrt((px - wx) ** 2 + (py - wy) ** 2)
            
            if current_iou > max_iou and distance < 100: # Increased distance threshold slightly
                max_iou = current_iou
                best_match_person_data = person
        
        weapon_person_matches[i] = best_match_person_data

    return weapon_person_matches


# --- Persistent Weapon Carrier Tracking & ReID ---
json_file = "weapon_carrier_embeddings.json"  # File to store embeddings
REID_SIMILARITY_THRESHOLD = 0.2 # INCREASED FOR ROBUSTNESS

# Server configuration
SERVER_URL = "http://localhost:8000"

# Replace load_saved_embeddings with API call
def load_saved_embeddings():
    try:
        response = requests.get(f"{SERVER_URL}/dump")
        response.raise_for_status()
        data = response.json()
        return {int(k): np.array(v) for k, v in data.items()}
    except Exception as e:
        logging.warning(f"Failed to load embeddings from server: {e}")
        return {}

# --- Refined Embedding Management ---
# Cache embeddings locally to avoid frequent HTTP calls
saved_embeddings = load_saved_embeddings()
cache_updated = False  # Flag to track if the cache needs updating

# Update the local cache of embeddings from the server.
def update_local_embeddings(force_refresh=False):
    """Update the local cache of embeddings from the server."""
    global saved_embeddings, cache_updated
    if force_refresh or cache_updated:
        try:
            new_embeddings = load_saved_embeddings()
            saved_embeddings.update(new_embeddings)
            cache_updated = False  # Reset the flag after updating
            logging.info("Local embeddings cache updated.")
        except Exception as e:
            logging.warning(f"Failed to update local embeddings cache: {e}")

# --- Optimized Embedding Upload Management ---
# Cache embeddings locally to avoid frequent uploads
pending_embeddings = []  # List to store embeddings to be uploaded
upload_interval = 30  # Number of frames between uploads
frame_counter = 0  # Counter to track frames

def upload_pending_embeddings():
    """Upload pending embeddings to the server."""
    global pending_embeddings
    if not pending_embeddings:
        return
    try:
        payload = {str(wid): embedding.tolist() for wid, embedding in pending_embeddings}
        response = requests.post(f"{SERVER_URL}/append", json=payload)
        response.raise_for_status()
        logging.info(f"Uploaded {len(pending_embeddings)} embeddings to server.")
        pending_embeddings.clear()  # Clear the pending list after successful upload
    except Exception as e:
        logging.warning(f"Failed to upload embeddings to server: {e}")

# Modify save_embedding to add embeddings to the pending list
def save_embedding(embedding, wid):
    try:
        pending_embeddings.append((wid, embedding))
        logging.info(f"Queued embedding for ID {wid} for upload.")
    except Exception as e:
        logging.warning(f"Failed to queue embedding for upload: {e}")

# Call upload_pending_embeddings periodically
frame_counter += 1
if frame_counter % upload_interval == 0:
    upload_pending_embeddings()

# Call update_local_embeddings only when necessary
# Replace this line:
# update_local_embeddings()
# With:
update_local_embeddings(force_refresh=False)

while True:
    ret, frame = cap.read()
    if not ret:
        logging.error("Failed to capture frame")
        break

    annotated_frame = frame.copy()

    # --- 1. Detect Guns ---
    gun_results = gun_model(frame, conf=0.35)
    gun_boxes = [] # List of (xyxy, conf)
    gun_boxes_xyxy = [] # List of xyxy numpy arrays
    if gun_results[0].boxes is not None:
        for box, cls, conf in zip(gun_results[0].boxes.xyxy, gun_results[0].boxes.cls, gun_results[0].boxes.conf):
            if gun_model.names[int(cls)] == "guns":
                box_np = box.cpu().numpy()
                gun_boxes.append((box_np, conf))
                gun_boxes_xyxy.append(box_np)
                
                wx1, wy1, wx2, wy2 = map(int, box_np)
                cv2.rectangle(annotated_frame, (wx1, wy1), (wx2, wy2), (255, 0, 0), 2)
                cv2.putText(
                    annotated_frame,
                    f"Gun ({conf:.2f})",
                    (wx1, wy1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2
                )
    logging.debug(f"Gun detections: {len(gun_boxes)} boxes")

    # --- 2. Detect People and Calculate Embeddings (Unified Loop) ---
    person_data = detect_people(frame, person_model, conf_threshold=0.4)
    
    # --- 3. Re-Identify All People Against Saved Weapon Carriers ---
    for person in person_data:
        x1, y1, x2, y2 = map(int, person['box'])
        current_embedding = person['embedding']
        
        # Check against saved weapon carrier embeddings
        best_match_id = None
        max_similarity = 0
        
        for saved_id, embedding_list in saved_embeddings.items():
            for saved_embedding in embedding_list:
                similarity = cosine_similarity([current_embedding], [saved_embedding])[0][0]
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_match_id = saved_id

        # Determine lock status and draw boxes for ALL people
        if best_match_id is not None and max_similarity > REID_SIMILARITY_THRESHOLD:
            # Person is a previously known weapon carrier
            person['reid_locked_id'] = best_match_id
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(
                annotated_frame, 
                f"Weapon Carrier: ID {best_match_id} ({max_similarity:.2f})", 
                (x1, y1 - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2
            )
        else:
            # Not a locked carrier (or similarity too low)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                annotated_frame, 
                f"Person ID: {person['pid']}", 
                (x1, y1 - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )


    # --- 4. Lock onto New Weapon Carriers (Gun-Person Match) ---
    weapon_person_matches = match_people_to_guns(person_data, gun_boxes_xyxy)
    
    for gun_index, weapon_person_match in enumerate(weapon_person_matches):
        if weapon_person_match is not None:
            # This person is currently holding a gun/is near a gun
            weapon_person_match['is_weapon_carrier'] = True
            
            # Use the existing ID or the newly generated general ID
            carrier_id = weapon_person_match['reid_locked_id'] 
            
            # If re-ID didn't lock on (it's a new weapon carrier)
            if carrier_id is None:
                carrier_id = weapon_person_match['pid'] # Use the general person ID
                logging.debug(f"New potential weapon carrier detected. Assigning new ID: {carrier_id}")
            
            x1, y1, x2, y2 = map(int, weapon_person_match['box'])
            current_embedding = weapon_person_match['embedding']
            
            # SAVE/UPDATE EMBEDDING (Enrollment)
            # This function handles the logic of only saving if the embedding is significantly different
            save_embedding(current_embedding, carrier_id)
            
            # Re-draw the box with the final, confirmed weapon carrier ID
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 4) # Thicker red box
            cv2.putText(
                annotated_frame, 
                f"WEAPON LOCKED ID: {carrier_id}", 
                (x1, y2 + 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3
            )
        
    # --- 5. Reload saved embeddings after potential saving
    update_local_embeddings(force_refresh=False)

    # Display the frame
    cv2.imshow("Weapon Detection + Robust ReID", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
