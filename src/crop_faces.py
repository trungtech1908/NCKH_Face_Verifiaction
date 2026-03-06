import cv2
import os
from face_detector import FaceDetector

detector = FaceDetector(
    "external/retinaface/weights/mobilenet0.25_Final.pth"
)

img = cv2.imread("data/test.jpg")

faces = detector.detect_and_crop(img)

print("Detected faces:", len(faces))

os.makedirs("data/faces", exist_ok=True)

for i, face in enumerate(faces):

    face_img = face["face"]

    path = f"data/faces/face_{i}.jpg"

    cv2.imwrite(path, face_img)

    print("Saved:", path)