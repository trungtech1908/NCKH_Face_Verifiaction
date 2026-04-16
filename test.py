import cv2
import time
import numpy as np
from qdrant_client import QdrantClient
from dotenv import load_dotenv
import os
from collections import Counter
from insightface.app import FaceAnalysis

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
app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
app.prepare(ctx_id=0)

# ===== Hiển thị đang chạy CPU hay GPU =====
def _infer_runtime_device(face_app) -> str:
    """
    Best-effort: đọc providers thực tế từ onnxruntime session bên trong insightface.
    """
    try:
        models = getattr(face_app, "models", None)
        if isinstance(models, dict) and models:
            for m in models.values():
                sess = getattr(m, "session", None)
                if sess is not None and hasattr(sess, "get_providers"):
                    providers = sess.get_providers()
                    return "GPU (CUDA)" if any("CUDA" in p for p in providers) else "CPU"
    except Exception:
        pass
    return "Unknown"

device_label = _infer_runtime_device(app)
print(f"[InsightFace] Runtime device: {device_label}")

# ================== CAMERA ==================
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

# ================== BIẾN ==================
prev_time = 0
last_embedding_time = 0

# lưu kết quả cho từng face (theo index)
face_results = {}

# ================== LOOP ==================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))

    faces = app.get(frame)

    # ===== 3s mới update embedding =====
    if time.time() - last_embedding_time > 0.5:
        face_results = {}

        for i, face in enumerate(faces):
            embedding = face.embedding
            embedding = embedding / np.linalg.norm(embedding)
            query_vector = embedding.tolist()

            results = client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector,
                limit=15
            )

            name = "Unknown"
            best_score = None

            if results.points:
                # Unknown cũng hiển thị score cao nhất đo được
                try:
                    best_score = max(p.score for p in results.points)
                except Exception:
                    best_score = None

                valid_points = [p for p in results.points if p.score > 0.6]

                if valid_points:
                    usernames = [
                        p.payload.get("username", "Unknown")
                        for p in valid_points
                    ]

                    counter = Counter(usernames)
                    most_common, count = counter.most_common(1)[0]

                    if count >= 2:
                        name = most_common
                        scores = [p.score for p in valid_points if p.payload.get("username", "Unknown") == most_common]
                        if scores:
                            best_score = sum(scores) / len(scores)

            face_results[i] = (name, best_score)

        last_embedding_time = time.time()

    # ===== VẼ =====
    for i, face in enumerate(faces):
        x1, y1, x2, y2 = map(int, face.bbox)

        name, score = face_results.get(i, ("Unknown", None))
        is_unknown = str(name).strip().lower() in {"unknown", "unknow"}
        color = (0, 0, 255) if is_unknown else (0, 255, 0)  # đỏ nếu Unknown

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = name
        if score is not None:
            label = f"{name} ({score:.2f})".replace(".", ",")
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # ===== FPS =====
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time

    cv2.putText(frame, f"FPS: {int(fps)}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"Device: {device_label}", (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()