from ultralytics import YOLO
import cv2
import numpy as np
import person_recognition
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Cosine similarity function
def cosine_similarity(emb1, emb2):
    try:
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    except Exception as e:
        logger.error(f"Error in cosine similarity: {e}")
        return -1

# YOLO model - revert to yolo11n.pt for debugging (lighter model)
model = YOLO("yolo11n.pt")  # Try 'yolo11m.pt' again after confirming setup
device = "cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"
logger.info(f"Using device: {device}")

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    logger.error("Cannot open webcam")
    exit()
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Tracking variables
reference_embedding = None
person_id = "1"  # Fixed ID for the first person
similarity_threshold = 0.7  # Lowered for debugging; adjust based on results

while True:
    ret, frame = cap.read()
    if not ret:
        logger.error("Failed to read frame from webcam")
        break

    # YOLO detection
    try:
        results = model(frame, device=device)
        annotated_frame = results[0].plot().copy()
        logger.debug(f"Detected {len(results[0].boxes)} objects")
    except Exception as e:
        logger.error(f"YOLO detection failed: {e}")
        annotated_frame = frame.copy()  # Fallback to show raw frame

    # If no reference yet, set the first detected person as reference
    if reference_embedding is None and len(results[0].boxes) > 0:
        box = results[0].boxes[0]  # Take first detection
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        person_roi = frame[y1:y2, x1:x2]
        try:
            person_roi_resized = cv2.resize(person_roi, (128, 256))
            reference_embedding = person_recognition.get_embedding(person_roi_resized)
            logger.info("Set reference embedding for first person")
            # Draw ID
            cv2.putText(
                annotated_frame,
                f"Person ID: {person_id}",
                (x1 + 50, y1 - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )
        except Exception as e:
            logger.error(f"Failed to get embedding: {e}")

    # For subsequent frames, find the best matching person
    elif reference_embedding is not None:
        best_match_box = None
        best_similarity = -1
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            person_roi = frame[y1:y2, x1:x2]
            try:
                person_roi_resized = cv2.resize(person_roi, (128, 256))
                embedding = person_recognition.get_embedding(person_roi_resized)
                similarity = cosine_similarity(reference_embedding, embedding)
                logger.debug(f"Similarity score: {similarity}")
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_box = (x1, y1, x2, y2)
            except Exception as e:
                logger.error(f"Embedding or similarity failed: {e}")

        # If a good match is found, annotate it
        if best_match_box and best_similarity > similarity_threshold:
            x1, y1, _, _ = best_match_box
            cv2.putText(
                annotated_frame,
                f"Person ID: {person_id} (Sim: {best_similarity:.2f})",
                (x1 + 50, y1 - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )
            logger.info(f"Matched person with similarity {best_similarity:.2f}")
        else:
            logger.debug("No match found above threshold")

    # Show frame
    try:
        cv2.imshow("YOLO", annotated_frame)
    except Exception as e:
        logger.error(f"Display failed: {e}")

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
