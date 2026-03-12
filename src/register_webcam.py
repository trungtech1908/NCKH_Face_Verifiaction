import cv2
import os
import time
from face_detector import FaceDetector

DATASET_DIR = "data/faces"
TOTAL_IMAGES = 40
CAPTURE_DELAY = 0.4
MARGIN = 40


def create_new_user_folder(base_path):

    os.makedirs(base_path, exist_ok=True)

    users = [d for d in os.listdir(base_path) if d.startswith("user_")]

    if not users:
        new_id = 1
    else:
        ids = [int(u.split("_")[1]) for u in users]
        new_id = max(ids) + 1

    user_folder = os.path.join(base_path, f"user_{new_id}")
    os.makedirs(user_folder)

    return user_folder, new_id


def blur_score(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def get_instruction(count):

    instructions = [
        "LOOK CENTER",
        "TURN LEFT",
        "TURN RIGHT",
        "LOOK UP",
        "LOOK DOWN"
    ]

    step = count // 8
    step = min(step, len(instructions) - 1)

    return instructions[step]


def main():

    save_dir, user_id = create_new_user_folder(DATASET_DIR)

    print("Registering user:", user_id)
    print("Saving images to:", save_dir)

    detector = FaceDetector(
        "external/retinaface/weights/mobilenet0.25_Final.pth"
    )

    cap = cv2.VideoCapture(0)

    count = 0
    last_capture = time.time()

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()

        faces = detector.detect(frame)

        for face in faces:

            x1, y1, x2, y2 = face["box"]

            x1 = max(0, x1 - MARGIN)
            y1 = max(0, y1 - MARGIN)

            x2 = min(frame.shape[1], x2 + MARGIN)
            y2 = min(frame.shape[0], y2 + MARGIN)

            cv2.rectangle(display, (x1,y1),(x2,y2),(0,255,0),2)

            face_img = frame[y1:y2, x1:x2]

            if face_img.size == 0:
                continue

            face_img = cv2.resize(face_img,(112,112))

            if time.time() - last_capture > CAPTURE_DELAY:

                if blur_score(face_img) > 100:

                    save_path = os.path.join(
                        save_dir,
                        f"{count}.jpg"
                    )

                    cv2.imwrite(save_path, face_img)

                    print("Saved:", save_path)

                    count += 1
                    last_capture = time.time()

        instruction = get_instruction(count)

        cv2.putText(display,
                    f"User: user_{user_id}",
                    (20,30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255,255,0),
                    2)

        cv2.putText(display,
                    f"Images: {count}/{TOTAL_IMAGES}",
                    (20,70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255,255,0),
                    2)

        cv2.putText(display,
                    instruction,
                    (20,110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0,255,0),
                    2)

        cv2.imshow("Face Registration", display)

        if count >= TOTAL_IMAGES:
            break

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    print("Registration complete!")


if __name__ == "__main__":
    main()