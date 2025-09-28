import cv2
from ultralytics import YOLO
    
# Load class names from file
# cap = cv2.VideoCapture('doorinout.mp4')

class_list = ["guns"]

# Initialize the YOLO model
model = YOLO("best3.pt")
print(model.names)


cap = cv2.VideoCapture(0)
# frame_width = int(cap.get(3))
# frame_height = int(cap.get(4))
# output_video = cv2.VideoWriter('11.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (frame_width, frame_height))
enter_area = [(180, 430), (200, 460), (480, 440), (480, 400)]
exit_area = [(210, 480), (230, 520), (495, 495), (495, 465)]

people_entered = set()
counter_enter = 0
count =0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    count += 1
    if count % 3 != 0:
        continue
    results = model.track(frame, conf=0.3)
    if results[0].boxes is None:
        continue

    boxes = results[0].boxes.data
    if results[0].boxes.id is not None:
        ids = results[0].boxes.id
    else:
        ids = 0
    class_ids = results[0].boxes.cls

    list_value = []
    for box, class_ids, class_id in zip(boxes, class_ids, class_ids):
        x1, y1, x2, y2 = map(int, box[:4].tolist())
        list_value.append([x1, y1, x2, y2, class_ids])

    bbox_id = list_value

    for bbox in bbox_id:
        x3, y3, x4, y4, class_ids = bbox
        
        cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)    
        text_position = (x3, y3 - 10)
        if class_list[int(class_id)] == "guns":
            text = "Weapone detected"
        else:
            text = "False detection"
        cv2.putText(frame, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
    # Show the frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Prclearess 'ESC' to exit
        break

cap.release()
# output_video.release()
cv2.destroyAllWindows()
