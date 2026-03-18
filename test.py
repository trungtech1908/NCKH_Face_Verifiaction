import cv2
import time
import numpy as np
from insightface.app import FaceAnalysis
from qdrant_client import QdrantClient
from dotenv import load_dotenv
import os
from collections import Counter

# ================== LOAD ENV ==================
load_dotenv()
URL_QDRANT = os.getenv("URL_QDRANT")
API_QDRANT = os.getenv("API_QDRANT")

# ================== QDRANT ==================
client = QdrantClient(
    url=URL_QDRANT,
    api_key=API_QDRANT
)

COLLECTION_NAME = "NCKH_Face_Verification"

# ================== INSIGHTFACE ==================
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0)  # GPU, nếu lỗi thì dùng -1

# ================== WEBCAM ==================
cap = cv2.VideoCapture(0)

# ================== BIẾN ==================
prev_time = 0
last_embedding_time = 0
current_name = "Unknown"

# ================== LOOP ==================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # resize tăng FPS
    frame = cv2.resize(frame, (640, 480))

    # ===== detect face =====
    faces = app.get(frame)

    for face in faces:
        x1, y1, x2, y2 = map(int, face.bbox)

        # ===== 3s mới embedding + search =====
        if time.time() - last_embedding_time > 3:
            embedding = face.embedding
            embedding = embedding / np.linalg.norm(embedding)
            query_vector = embedding.tolist()

            # ===== query Qdrant =====
            results = client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector,
                limit=15
            )

            if results.points:
                # lọc theo threshold
                valid_points = [p for p in results.points if p.score > 0.5]

                if valid_points:
                    usernames = [
                        p.payload.get("username", "Unknown")
                        for p in valid_points
                    ]

                    counter = Counter(usernames)
                    most_common, count = counter.most_common(1)[0]

                    # optional: điều kiện chắc chắn hơn
                    if count >= 2:
                        current_name = most_common
                    else:
                        current_name = "Unknown"
                else:
                    current_name = "Unknown"
            else:
                current_name = "Unknown"

            last_embedding_time = time.time()

        # ===== vẽ bbox + label =====
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, current_name, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # ===== FPS =====
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time

    cv2.putText(frame, f"FPS: {int(fps)}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # ===== SHOW =====
    cv2.imshow("Face Recognition", frame)

    # ESC để thoát
    if cv2.waitKey(1) & 0xFF == 27:
        break

# ================== CLEAN ==================
cap.release()
cv2.destroyAllWindows()