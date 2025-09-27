# gunmatch.py
import numpy as np

# ---------- geometry helpers ----------
def _center_xyxy(b):
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

def dedup_boxes(boxes, iou_thr=0.95):
    """Remove near-duplicate boxes (e.g., when same person matched to two guns)."""
    uniq = []
    for b in boxes:
        if b is None:
            continue
        if not any(iou_xyxy(b, u) > iou_thr for u in uniq):
            uniq.append(b)
    return uniq

# ---------- detection ----------
def detect_guns(frame, gun_model, conf=0.20, imgsz=640):
    """
    Returns: list of [x1,y1,x2,y2] (floats) for guns.
    Assumes gun_model was trained gun-only, so every box is a gun.
    """
    r = gun_model.predict(frame, conf=conf, imgsz=imgsz, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return []
    return r.boxes.xyxy.cpu().numpy().tolist()

def detect_people(frame, person_model, conf=0.20, imgsz=640):
    """
    Returns: list of [x1,y1,x2,y2] (floats) for persons.
    Uses COCO model: class 0 == 'person'.
    """
    r = person_model.predict(frame, conf=conf, imgsz=imgsz, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return []
    xyxy = r.boxes.xyxy.cpu().numpy()
    cls  = r.boxes.cls.cpu().numpy().astype(int)
    return xyxy[cls == 0].tolist()

# ---------- matching ----------
def match_people_to_guns(person_boxes, gun_boxes):
    """
    For each gun box, choose the person with:
      - max IoU if any overlap, else
      - smallest center distance.
    Returns list of person boxes (or None), one per gun (same order as gun_boxes).
    """
    if not gun_boxes:
        return []
    if not person_boxes:
        return [None] * len(gun_boxes)

    p_centers = np.array([_center_xyxy(p) for p in person_boxes])
    matched = []
    for g in gun_boxes:
        ious = np.array([iou_xyxy(g, p) for p in person_boxes])
        if (ious > 0).any():
            j = int(np.argmax(ious))
            matched.append(person_boxes[j])
        else:
            g_center = _center_xyxy(g)
            dists = np.linalg.norm(p_centers - g_center, axis=1)
            j = int(np.argmin(dists))
            matched.append(person_boxes[j])
    return matched

def pairs_for_frame(frame, gun_model, person_model, conf_gun=0.20, conf_person=0.20, imgsz=640):
    """
    Convenience wrapper:
      - detects guns & persons
      - computes matches
    Returns: (gun_boxes, matched_person_boxes, person_boxes_all)
    """
    guns = detect_guns(frame, gun_model, conf=conf_gun, imgsz=imgsz)
    people = detect_people(frame, person_model, conf=conf_person, imgsz=imgsz)
    matches = match_people_to_guns(people, guns)
    return guns, matches, people
