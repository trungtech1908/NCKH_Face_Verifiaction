import cv2
from face_detector import FaceDetector

detector = FaceDetector(
    "external/retinaface/weights/mobilenet0.25_Final.pth"
)

img = cv2.imread("data/test.jpg")

boxes = detector.detect(img)

print("Faces:", len(boxes))

for b in boxes:
    x1,y1,x2,y2,score = b

    cv2.rectangle(
        img,
        (int(x1),int(y1)),
        (int(x2),int(y2)),
        (0,255,0),
        2
    )

cv2.imwrite("data/output.jpg", img)