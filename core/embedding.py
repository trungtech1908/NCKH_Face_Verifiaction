import cv2
import time
import numpy as np
from qdrant_client import QdrantClient
from dotenv import load_dotenv
import os
from collections import Counter

# ===== YOUR EMBEDDER =====
from core.embedding import FaceEmbedder

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

# ================== EMBEDDER ==================
embedder = FaceEmbedder()

# ================== CAMERA ==================
print("Starting camera...")
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

if not cap.isOpened():
    print("❌ Cannot open camera")
    exit()

print("✅ Camera opened")

# ================== BIẾN ==================
prev_time = 0
last_embedding_time = 0
current_name = "Unknown"

# ================== LOOP ==================
while True:
    ret, frame = cap.read()

    if not ret:
        print("❌ Failed to read frame")
        break

    # resize tăng FPS
    frame = cv2.resize(frame, (640, 480))

    # ===== EMBEDDING mỗi 3 giây =====
    if time.time() - last_embedding_time > 3:
        print("🔍 Running embedding...")

        embedding = embedder.extract(frame)

        if embedding is None:
            print("⚠️ No face detected")
            current_name = "Unknown"
        else:
            print("✅ Embedding OK:", embedding.shape)

            embedding = embedding / np.linalg.norm(embedding)
            query_vector = embedding.tolist()

            # ===== QUERY QDRANT =====
            results = client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector,
                limit=15
            )

            if results.points:
                valid_points = [p for p in results.points if p.score > 0.5]

                if valid_points:
                    usernames = [
                        p.payload.get("username", "Unknown")
                        for p in valid_points
                    ]

                    counter = Counter(usernames)
                    most_common, count = counter.most_common(1)[0]

                    if count >= 2:
                        current_name = most_common
                        print(f"🎯 Match: {current_name} ({count} votes)")
                    else:
                        current_name = "Unknown"
                        print("⚠️ Not confident")
                else:
                    current_name = "Unknown"
                    print("⚠️ No valid match")
            else:
                current_name = "Unknown"
                print("⚠️ No results from Qdrant")

        last_embedding_time = time.time()

    # ===== HIỂN THỊ =====
    cv2.putText(frame, current_name, (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    # ===== FPS =====
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time

    cv2.putText(frame, f"FPS: {int(fps)}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Face Recognition", frame)

    # ESC để thoát
    if cv2.waitKey(1) & 0xFF == 27:
        break

# ================== CLEAN ==================
cap.release()
cv2.destroyAllWindows()