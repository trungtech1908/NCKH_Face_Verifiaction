import cv2
import time
import numpy as np
import threading
import os
from collections import Counter
from qdrant_client import QdrantClient
from dotenv import load_dotenv
from insightface.app import FaceAnalysis
from src.utility import get_crop_face
from core.anti_spoof import AntiSpoof

# ================== SETUP ==================
load_dotenv()
fas = AntiSpoof(model_dir="./external/anti_spoofing")
client = QdrantClient(url=os.getenv("URL_QDRANT"), api_key=os.getenv("API_QDRANT"))
COLLECTION_NAME = "NCKH_Face_Verification"

app = FaceAnalysis(name="buffalo_l", allowed_modules=['detection', 'recognition'])
app.prepare(ctx_id=0)

# ================== BIẾN HỆ THỐNG ==================
prev_time, last_embedding_time = 0, 0
next_id = 0
face_results, face_statuses, last_face_centers = {}, {}, {}

def query_qdrant_async(faces_data):
    global face_results
    for f_id, emb in faces_data:
        res = client.query_points(collection_name=COLLECTION_NAME, query=emb.tolist(), limit=5)
        
        # DEBUG: In ra để xem score thực tế là bao nhiêu
        if res.points:
            print(f"--- QDRANT RAW --- ID: {f_id} | Top 1 Score: {res.points[0].score:.4f} | Name: {res.points[0].payload.get('username')}")

        name = "Unknown"
        # Nâng ngưỡng từ 0.5 cũ lên 0.7 (đây là mức "vàng" của InsightFace)
        valid = [p.payload.get("username", "Unknown") for p in res.points if p.score > 0.7] 
        
        if valid:
            name, count = Counter(valid).most_common(1)[0]
            if count < 2: name = "Unknown"
        face_results[f_id] = name

# ================== MAIN LOOP ==================
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.resize(frame, (640, 480))
    faces = app.get(frame)
    
    curr_ids, new_centers = [], {}

    # 1. TRACKING ID (Theo dõi tâm mặt)
    for face in faces:
        x1, y1, x2, y2 = map(int, face.bbox)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        matched_id = None
        for fid, l_center in last_face_centers.items():
            if np.sqrt((cx-l_center[0])**2 + (cy-l_center[1])**2) < 60:
                matched_id = fid
                break
            
        if matched_id is None:
            matched_id = next_id
            next_id += 1
        curr_ids.append(matched_id)
        new_centers[matched_id] = (cx, cy)
    last_face_centers = new_centers.copy()

    # 2. FAS & RECOGNITION (3 giây/lần)
    if time.time() - last_embedding_time > 3:
        to_query = []
        for face, fid in zip(faces, curr_ids):
            i27 = get_crop_face(frame, face.bbox, 2.7)
            i40 = get_crop_face(frame, face.bbox, 4.0)
            label, score = fas.predict(i27, i40)
            print(f"--- FAS CHECK --- ID: {fid} | Label: {label} | Score: {score:.4f}")
            cv2.imshow(f"FAS_27_Debug", i27)
            cv2.imshow(f"FAS_40_Debug", i40)
            face_statuses[fid] = "Real" if (label == 1 and score > 0.9) else "Fake"
            if face_statuses[fid] == "Real":
                to_query.append((fid, face.embedding / np.linalg.norm(face.embedding)))
        
        if to_query: threading.Thread(target=query_qdrant_async, args=(to_query,)).start()
        last_embedding_time = time.time()

    # 3. VẼ UI
    for face, fid in zip(faces, curr_ids):
        x1, y1, x2, y2 = map(int, face.bbox)
        name = face_results.get(fid, "Searching...")
        status = face_statuses.get(fid, "Scanning...")
        
        color = (255, 255, 255) # Trắng
        if status == "Real": color = (0, 255, 0) # Xanh
        elif status == "Fake": color = (0, 0, 255) # Đỏ

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{name}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"STATUS: {status}", (x1, y2+25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 4. FPS
    fps = 1 / (time.time() - prev_time) if prev_time else 0
    prev_time = time.time()
    cv2.putText(frame, f"FPS: {int(fps)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    cv2.imshow("NCKH - Face Verification", frame)
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()