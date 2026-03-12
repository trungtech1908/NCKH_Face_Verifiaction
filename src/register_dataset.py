import cv2
import os
from face_detector import FaceDetector

VIDEO_DIR = "data/video_faces"
OUTPUT_DIR = "data/faces"

FRAME_SKIP = 5
MARGIN = 40

detector = FaceDetector(
    "external/retinaface/weights/mobilenet0.25_Final.pth"
)


def process_video(video_path):

    video_name = os.path.splitext(os.path.basename(video_path))[0]

    save_dir = os.path.join(
        OUTPUT_DIR,
        f"video_dataset_{video_name}"
    )

    os.makedirs(save_dir, exist_ok=True)

    print("Processing:", video_name)

    cap = cv2.VideoCapture(video_path)

    frame_id = 0
    count = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # skip frames
        if frame_id % FRAME_SKIP != 0:
            frame_id += 1
            continue

        faces = detector.detect(frame)

        for face in faces:

            x1, y1, x2, y2 = face["box"]

            # padding
            x1 = max(0, x1 - MARGIN)
            y1 = max(0, y1 - MARGIN)

            x2 = min(frame.shape[1], x2 + MARGIN)
            y2 = min(frame.shape[0], y2 + MARGIN)

            face_img = frame[y1:y2, x1:x2]

            if face_img.size == 0:
                continue

            face_img = cv2.resize(face_img, (112,112))

            save_path = os.path.join(
                save_dir,
                f"{count}.jpg"
            )

            cv2.imwrite(save_path, face_img)

            count += 1

        frame_id += 1

    cap.release()

    print("Saved", count, "faces for", video_name)
    print()


def main():

    videos = [
        f for f in os.listdir(VIDEO_DIR)
        if f.endswith(".mp4")
    ]

    if not videos:
        print("No videos found!")
        return

    for video in videos:

        video_path = os.path.join(VIDEO_DIR, video)

        process_video(video_path)


if __name__ == "__main__":
    main()