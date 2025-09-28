import os

import cv2


def save_clipped(image, bbox, person_id, frame_index):
    save_path = "suspect_images/"
    """
    Clips and saves a bounding box region from the image.

    image: numpy array (BGR frame from cv2)
    bbox: (x1, y1, x2, y2) coordinates of the bounding box
    save_path: directory to save images
    person_id: unique id for the person
    """
    x1, y1, x2, y2 = bbox

    # clip coordinates to image boundaries
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    # crop
    cropped = image[y1:y2, x1:x2]

    # make folder if not exists
    os.makedirs(save_path, exist_ok=True)

    # filename with id + frame
    filename = os.path.join(save_path, f"person_{person_id}-{frame_index}.jpg")
    print(f"Saved file at {filename}")
    cv2.imwrite(filename, cropped)

    return filename