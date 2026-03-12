import cv2
import time
from face_detector import FaceDetector

detector = FaceDetector(
    "external/retinaface/weights/mobilenet0.25_Final.pth"
)
cap = cv2.VideoCapture("test_video/test02.mp4")
prev_time = time.time()
while True:

    ret, frame = cap.read()
    if not ret:
        break

    start = time.time()

    faces = detector.detect(frame)

    for face in faces:
        x1,y1,x2,y2 = face["box"]
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

    end = time.time()

    fps = 1 / (end - start)

    cv2.putText(
        frame,
        f"FPS: {fps:.2f}",
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()