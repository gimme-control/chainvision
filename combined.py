import cv2
from ultralytics import YOLO
import time

# -------- CONFIG --------
USE_GPU = True
PERSON_MODEL = "yolov8n.pt"   # for people/tracking
WEAPON_MODEL = "best3.pt"     # your custom gun/knife model
PERSON_CONF = 0.3
WEAPON_CONF = 0.3
FRAME_SKIP = 1  # set to 2 or 3 if still slow

# -------- Init Models --------
device = "cuda" if USE_GPU else "cpu"
print("Using device:", device)

person_model = YOLO(PERSON_MODEL)
weapon_model = YOLO(WEAPON_MODEL)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # lower resolution for extra FPS
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_index = 0
fps_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_index += 1
    if frame_index % FRAME_SKIP != 0:
        continue

    # ---- PERSON DETECTION/TRACKING ----
    try:
        person_results = person_model.track(
            frame, device=device, persist=True, conf=PERSON_CONF
        )
    except:
        person_results = person_model.predict(
            frame, device=device, conf=PERSON_CONF
        )

    pres = person_results[0]
    annotated = frame.copy()
    person_boxes = []
    person_ids = []

    if pres.boxes is not None:
        boxes_data = pres.boxes.data.tolist()
        try:
            ids = pres.boxes.id.cpu().numpy().tolist()
        except:
            ids = [None] * len(boxes_data)

        for i, row in enumerate(boxes_data):
            x1, y1, x2, y2 = map(int, row[:4])
            pid = ids[i]
            person_boxes.append((x1, y1, x2, y2))
            person_ids.append(pid)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (50, 200, 50), 2)
            label = "Person"
            if pid is not None:
                label += f" ID:{pid}"
            cv2.putText(annotated, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 200, 50), 2)

    # ---- WEAPON DETECTION ----
    wres = weapon_model.predict(frame, device=device, conf=WEAPON_CONF)[0]

    if wres.boxes is not None:
        for box_row, cls_idx, conf in zip(
            wres.boxes.data.tolist(), wres.boxes.cls.tolist(), wres.boxes.conf.tolist()
        ):
            x1, y1, x2, y2 = map(int, box_row[:4])
            # Get name from model
            try:
                name = wres.names[int(cls_idx)]
            except:
                name = str(int(cls_idx))

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(annotated, f"{name}:{conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # ---- FPS Display ----
    now = time.time()
    fps = 1.0 / (now - fps_time + 1e-9)
    fps_time = now
    cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.imshow("People + Weapon Detection", annotated)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
