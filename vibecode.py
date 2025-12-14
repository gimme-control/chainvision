import cv2
import numpy as np
from ultralytics import YOLO

# ---------------- Models ----------------
gun_model    = YOLO("best3.pt")        # your gun-only model (class_list = ["guns"])
person_model = YOLO("yolov8n.pt")      # COCO model; class 0 == "person"

# ---------------- Video -----------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# ---------------- Helpers ----------------
def center_xyxy(b):
    x1, y1, x2, y2 = b
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=float)

def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter + 1e-6
    return inter / ua

def match_people_to_guns(person_boxes, gun_boxes):
    """
    Returns list of person boxes (or None), one per gun box.
    Preference: max IoU>0; else nearest center distance.
    """
    if not gun_boxes:
        return []
    if not person_boxes:
        return [None] * len(gun_boxes)

    p_centers = np.array([center_xyxy(p) for p in person_boxes])
    matched = []
    for g in gun_boxes:
        # prefer overlap
        ious = np.array([iou_xyxy(g, p) for p in person_boxes])
        if (ious > 0).any():
            j = int(np.argmax(ious))
            matched.append(person_boxes[j])
            continue
        # else closest center
        g_center = center_xyxy(g)
        dists = np.linalg.norm(p_centers - g_center, axis=1)
        j = int(np.argmin(dists))
        matched.append(person_boxes[j])
    return matched

# ---------------- Loop ----------------
frame_skip = 3
count = 0

while True:
    ok, frame = cap.read()
    if not ok:
        break

    count += 1
    if count % frame_skip != 0:
        # still show the last processed frame to keep UI responsive
        cv2.imshow("Gun + Person pairing", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        continue

    # 1) Detect guns (your model). You used track(); predict() is fine here too.
    g_res = gun_model.predict(frame, conf=0.30, verbose=False)[0]
    gun_boxes = []
    if g_res.boxes is not None and len(g_res.boxes) > 0:
        g_xyxy = g_res.boxes.xyxy.cpu().numpy()
        # your model is gun-only, so every box is a gun:
        gun_boxes = [b for b in g_xyxy]

    # 2) Detect persons (class 0 in COCO)
    p_res = person_model.predict(frame, conf=0.25, verbose=False)[0]
    person_boxes = []
    if p_res.boxes is not None and len(p_res.boxes) > 0:
        p_xyxy = p_res.boxes.xyxy.cpu().numpy()
        p_cls  = p_res.boxes.cls.cpu().numpy().astype(int)
        person_boxes = [b for b, c in zip(p_xyxy, p_cls) if c == 0]  # 0 == 'person'

    # 3) Pair: for each gun, find the best person box (or None)
    matched_person_boxes = match_people_to_guns(person_boxes, gun_boxes)

    # 4) Draw overlays (red = gun, green = matched person)
    vis = frame.copy()
    for g_box, p_box in zip(gun_boxes, matched_person_boxes):
        gx1, gy1, gx2, gy2 = map(int, g_box)
        cv2.rectangle(vis, (gx1, gy1), (gx2, gy2), (0, 0, 255), 2)
        cv2.putText(vis, "GUN", (gx1, max(0, gy1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        if p_box is not None:
            px1, py1, px2, py2 = map(int, p_box)
            cv2.rectangle(vis, (px1, py1), (px2, py2), (0, 255, 0), 3)
            # optional: connect centers
            gc = (int((gx1 + gx2) / 2), int((gy1 + gy2) / 2))
            pc = (int((px1 + px2) / 2), int((py1 + py2) / 2))
            cv2.line(vis, gc, pc, (0, 255, 0), 2)
            cv2.putText(vis, "PERSON", (px1, max(0, py1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # (Optional) If you want the raw coordinates to use elsewhere:
    # matched_coords = [None if b is None else [int(v) for v in b] for b in matched_person_boxes]
    # print(matched_coords)

    cv2.imshow("Gun + Person pairing", vis)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
