import requests # Use requests directly for server communication
import gunmatch
from ultralytics import YOLO
import cv2
import image_utils
import person_recognition
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
import os
import logging
import atexit # To ensure we save on exit


# --- Server and Caching Configuration ---
SERVER_URL = "http://127.0.0.1:8000"
TIMEOUT = 5 # seconds for server requests
BATCH_UPLOAD_SIZE = 5  # Upload to server after every 5 new carriers are found
embedding_upload_cache = {} # Cache for new embeddings before uploading

picture_counter = 0

# --- Models ---
# Assuming these models are accessible in your environment
person_model = YOLO("models/yolov8n.pt")
gun_model = YOLO("models/best.pt")

# --- Logging ---
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Video ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 350)

# --- Persistent Weapon Carrier (Now managed by the server) ---
saved_embeddings = {} # This remains our fast, in-memory store for the session

# --- NEW: Load saved embeddings from the server using requests ---
try:
    logging.info(f"Fetching initial embeddings from server: {SERVER_URL}")
    response = requests.get(f"{SERVER_URL}/dump", timeout=TIMEOUT)
    response.raise_for_status() # Will raise an exception for bad status (4xx or 5xx)
    server_data = response.json()
    
    # The server stores a list for each key. We take the most recent (last) one.
    for key, value_list in server_data.items():
        if value_list: # Ensure the list is not empty
            # The key is the person ID, the value is their embedding
            saved_embeddings[int(key)] = np.array(value_list[-1])
    logging.info(f"Successfully loaded {len(saved_embeddings)} embeddings from the server.")
except requests.exceptions.RequestException as e:
    logging.error(f"Could not connect to or get data from the server at {SERVER_URL}. Error: {e}. Running with empty initial embeddings.")


def upload_cached_embeddings():
    """Submits the locally cached embeddings to the server and clears the cache."""
    global embedding_upload_cache
    if not embedding_upload_cache:
        return # Nothing to upload

    logging.info(f"Uploading batch of {len(embedding_upload_cache)} embeddings to the server...")
    try:
        # Use requests.post to send data to the /append endpoint
        response = requests.post(f"{SERVER_URL}/append", json=embedding_upload_cache, timeout=TIMEOUT)
        response.raise_for_status()
        
        logging.info(f"Server response: {response.json()}")
        embedding_upload_cache.clear() # Clear cache after successful upload
        logging.info("Upload successful, cache cleared.")
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to upload embeddings to the server at {SERVER_URL}. Error: {e}")

# Register the upload function to be called automatically on program exit
atexit.register(upload_cached_embeddings)


def cache_and_maybe_upload(embedding, weapon_carrier_id):
    """
    Saves a new weapon carrier's embedding to the in-memory store,
    caches it for upload, and triggers an upload if the batch size is reached.
    """
    global saved_embeddings, embedding_upload_cache
    
    # 1. Update the live, in-memory dictionary for immediate use
    saved_embeddings[weapon_carrier_id] = embedding

    # 2. Add to the upload cache, converting numpy array to a list for JSON compatibility
    #    The server key must be a string.
    embedding_upload_cache[str(weapon_carrier_id)] = embedding.tolist()
    logging.info(f"Cached new carrier ID {weapon_carrier_id}. Cache size: {len(embedding_upload_cache)}.")

    # 3. If the cache reaches the batch size, upload to the server
    if len(embedding_upload_cache) >= BATCH_UPLOAD_SIZE:
        upload_cached_embeddings()


# --- Thresholds ---
GUN_CONF_THRESHOLD = 0.4
PERSON_CONF_THRESHOLD = 0.7
REID_SIMILARITY_THRESHOLD = 0.7
IOU_THRESHOLD = 0.4
PROXIMITY_THRESHOLD = 150  # pixels

# --- Get frame dimensions ---
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# --- Define codec and create VideoWriter object ---
out = cv2.VideoWriter(
    "output_annotated.avi",
    cv2.VideoWriter_fourcc(*"XVID"),
    30,
    (frame_width, frame_height)
)

if not out.isOpened():
    logging.error("VideoWriter failed to open.")
    cap.release()
    cv2.destroyAllWindows()
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

    for person_box in person_boxes:
        x1, y1, x2, y2 = map(int, person_box)
        person_roi = frame[y1:y2, x1:x2]
        if person_roi.size == 0 or person_roi.shape[0] < 2 or person_roi.shape[1] < 2:
            continue

        person_roi_resized = cv2.resize(person_roi, (128, 256))
        embedding = person_recognition.get_embedding(person_roi_resized)

        # --- RE-IDENTIFICATION & TRACKING LOGIC ---
        
        is_known_weapon_carrier = False
        current_person_id = person_recognition.get_person_id(embedding) # Get temporary/session ID first

        # Check against the PERSISTENT (server-loaded) saved embeddings
        # We look for the best match against anyone ever saved.
        best_similarity = 0.0
        matched_saved_id = None
        
        for saved_id, saved_embedding in saved_embeddings.items():
            similarity = cosine_similarity([embedding], [saved_embedding])[0][0]
            if similarity > REID_SIMILARITY_THRESHOLD and similarity > best_similarity:
                best_similarity = similarity
                matched_saved_id = saved_id

        # If a match is found to a persistent carrier (even without a weapon present)
        if matched_saved_id is not None:
            is_known_weapon_carrier = True
            final_id = matched_saved_id # Use the persistent ID
            
            # 1. ALWAYS DRAW RED BOX FOR TRACKING KNOWN CARRIERS
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(annotated_frame, f"Tracking Carrier: ID {final_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            
            # 2. Picture logic (Kept separate from drawing)
            # You may want to refine this to track per-ID, but using the global counter for now:
            if picture_counter < 5:
                # Ensure you save pictures using the persistent ID
                image_utils.save_clipped_person(frame, (x1, y1, x2, y2), final_id, picture_counter)
                picture_counter += 1

        else:
            # If not a known persistent carrier, track with the session ID (green box)
            final_id = current_person_id
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Person ID: {final_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


    # --- Lock onto weapon carriers (Primary Detection & Persistence) ---
    for gun_box, matched_person_box in zip(gun_boxes, weapon_person_boxes):
        # Draw gun (BLUE)
        gx1, gy1, gx2, gy2 = map(int, gun_box)
        cv2.rectangle(annotated_frame, (gx1, gy1), (gx2, gy2), (255, 0, 0), 2)
        cv2.putText(annotated_frame, "Weapon", (gx1, gy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # If gun is matched to a person, confirm them as a carrier
        if matched_person_box is not None:
            px1, py1, px2, py2 = map(int, matched_person_box)
            person_roi = frame[py1:py2, px1:px2]
            if person_roi.size == 0 or person_roi.shape[0] < 2 or person_roi.shape[1] < 2:
                continue

            person_roi_resized = cv2.resize(person_roi, (128, 256))
            embedding = person_recognition.get_embedding(person_roi_resized)
            
            # *** USE THE EMBEDDING TO GET/CREATE THE ID FOR THIS NEW CARRIER ***
            weapon_id = person_recognition.get_person_id(embedding) 

            # Persist weapon carrier if new
            if weapon_id not in saved_embeddings:
                # This adds the new embedding to the in-memory cache and triggers server upload
                cache_and_maybe_upload(embedding, weapon_id)

            # Re-draw the carrier box in the PRIMARY CARRIER color (BRIGHT RED, Thicker line)
            # This overrides the initial tracking draw to signal the active threat
            cv2.rectangle(annotated_frame, (px1, py1), (px2, py2), (0, 0, 255), 4) 
            cv2.putText(annotated_frame, f"WEAPON CARRIER: ID {weapon_id}", (px1, py1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)


    out.write(annotated_frame)
    cv2.imshow("Weapon Detection and Tracking", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# --- Cleanup ---
# The atexit hook will automatically call upload_cached_embeddings(),
# but we can also call it here explicitly for clarity before shutdown.
logging.info("Exiting main loop. Performing final embedding upload...")
upload_cached_embeddings()

out.release()
cv2.destroyAllWindows()
logging.info("Video processing finished and resources released.")
image_utils.generate_summary(image_utils.image_list)