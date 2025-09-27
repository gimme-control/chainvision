import cv2
from flask import Flask, Response

app = Flask(__name__)
cap = cv2.VideoCapture(0)  # Default webcam

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

def gen():
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            print("Error: Failed to encode frame.")
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("Starting Flask server on http://127.0.0.1:5000/video_feed")
    app.run(host='0.0.0.0', port=5000, threaded=True)
