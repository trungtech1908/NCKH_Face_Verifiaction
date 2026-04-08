import cv2
import time
import numpy as np
from qdrant_client import QdrantClient
from dotenv import load_dotenv
import os
from collections import Counter
from insightface.app import FaceAnalysis
import threading # Thư viện để chạy đa luồng

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

# ================== DETECT + EMBEDDING ==================
# TỐI ƯU 1: Chỉ cho phép mô hình detection (tìm mặt) và recognition (trích xuất đặc trưng) chạy
app = FaceAnalysis(name="buffalo_l", allowed_modules=['detection', 'recognition'], providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0)

# ================== CAMERA ==================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# ================== BIẾN ==================
prev_time = 0
last_embedding_time = 0
face_results = {}

# TỐI ƯU 2: Hàm truy vấn Qdrant chạy ngầm để không làm đơ Camera
def query_qdrant_async(faces_data):
    global face_results
    temp_results = {}
    
    for i, embedding in faces_data:
        query_vector = embedding.tolist()
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=15
        )

        name = "Unknown"
        if results.points:
            valid_points = [p for p in results.points if p.score > 0.5]
            if valid_points:
                usernames = [p.payload.get("username", "Unknown") for p in valid_points]
                counter = Counter(usernames)
                most_common, count = counter.most_common(1)[0]
                if count >= 2:
                    name = most_common
        temp_results[i] = name
        
    face_results = temp_results # Cập nhật kết quả lên màn hình

# ================== LOOP ==================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))

    # AI chỉ chạy mô hình phát hiện & nhận diện (nhẹ hơn rất nhiều)
    faces = app.get(frame)

    # ===== 3s mới update embedding 1 lần bằng Thread =====
    if time.time() - last_embedding_time > 3:
        if len(faces) > 0:
            # Thu thập embedding của các khuôn mặt hiện tại
            faces_data = []
            for i, face in enumerate(faces):
                emb = face.embedding / np.linalg.norm(face.embedding)
                faces_data.append((i, emb))
            
            # Khởi chạy luồng ngầm để hỏi Qdrant (Main thread vẫn tiếp tục chạy camera)
            threading.Thread(target=query_qdrant_async, args=(faces_data,)).start()
            
        last_embedding_time = time.time()

    # ===== VẼ =====
    for i, face in enumerate(faces):
        x1, y1, x2, y2 = map(int, face.bbox)
        # Lấy tên từ kết quả của Thread (nếu chưa có thì để Unknown)
        name = face_results.get(i, "Unknown") 

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # ===== FPS =====
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time

    cv2.putText(frame, f"FPS: {int(fps)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()