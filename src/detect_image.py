import cv2
from face_detector import FaceDetector

detector = FaceDetector(
    "external/retinaface/weights/mobilenet0.25_Final.pth"
)

img = cv2.imread("data/test02.jpg")

boxes = detector.detect(img)

print("Faces:", len(boxes))
print("CHECK:", boxes)

for b in boxes:

    if isinstance(b, dict):
        x1, y1, x2, y2 = b["box"]
        score = b.get("score", 1.0)

    elif isinstance(b, (list, tuple)) and len(b) == 5:
        x1, y1, x2, y2, score = b

    elif isinstance(b, (list, tuple)) and len(b) == 2:
        box = b[0]
        x1, y1, x2, y2, score = box

    else:
        print("Unknown format:", b)
        continue

    cv2.rectangle(
        img,
        (int(x1), int(y1)),
        (int(x2), int(y2)),
        (0, 255, 0),
        2
    )

cv2.imwrite("data/output02.jpg", img)