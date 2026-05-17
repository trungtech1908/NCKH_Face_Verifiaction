import os
import cv2
from dotenv import load_dotenv
from collections import Counter
from insightface.app import FaceAnalysis
from qdrant_client import QdrantClient
from tqdm import tqdm

# ===== ENV =====
load_dotenv()
client = QdrantClient(
    url=os.getenv("URL_QDRANT"),
    api_key=os.getenv("API_QDRANT"),
    timeout=30
)

COLLECTION_NAME = "NCKH_Face_Verification"

# ===== MODEL =====
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0)
rec_model = app.models['recognition']

# ===== DATA =====
DATASET_PATH = "megaface-testsuite/megaface/data/megaface_testpack_v1.0/facescrub_images"

TOP_K = 15
THRESHOLD = 0.4
VOTE_K = 1

persons = sorted(os.listdir(DATASET_PATH))

# ===== SPLIT =====
first_70 = persons[:70]
last_10 = persons[-10:]

# ===== METRICS =====
total_known = 0
correct_known = 0

total_unknown = 0
correct_unknown = 0

# ================== 1. TEST KNOWN ==================
for person in tqdm(first_70, desc="KNOWN (70 persons)"):

    person_path = os.path.join(DATASET_PATH, person)
    if not os.path.isdir(person_path):
        continue

    images = sorted(os.listdir(person_path))
    test_images = images[20:]   # từ ảnh 21 trở đi

    for img_name in tqdm(test_images, desc=f"{person}", leave=False):

        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        # ===== giữ nguyên pipeline cũ =====
        face = cv2.resize(img, (112, 112))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        emb = rec_model.get_feat(face).flatten()

        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=emb.tolist(),
            limit=TOP_K
        )

        pred = "Unknown"

        if results.points:
            valid = [p for p in results.points if p.score > THRESHOLD]

            if valid:
                names = [p.payload.get("username", "Unknown") for p in valid]
                most_common, count = Counter(names).most_common(1)[0]

                if count >= VOTE_K:
                    pred = most_common

        total_known += 1
        if pred == person:
            correct_known += 1

# ================== 2. TEST UNKNOWN ==================
for person in tqdm(last_10, desc="UNKNOWN (10 persons)"):

    person_path = os.path.join(DATASET_PATH, person)
    if not os.path.isdir(person_path):
        continue

    images = sorted(os.listdir(person_path))

    for img_name in tqdm(images, desc=f"{person}", leave=False):

        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        face = cv2.resize(img, (112, 112))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        emb = rec_model.get_feat(face).flatten()

        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=emb.tolist(),
            limit=TOP_K
        )

        pred = "Unknown"

        if results.points:
            valid = [p for p in results.points if p.score > THRESHOLD]

            if valid:
                names = [p.payload.get("username", "Unknown") for p in valid]
                most_common, count = Counter(names).most_common(1)[0]

                if count >= VOTE_K:
                    pred = most_common

        total_unknown += 1
        if pred == "Unknown":
            correct_unknown += 1

# ================== RESULT ==================
print("\n===== RESULT =====")

print("\n--- KNOWN (70 persons) ---")
print("TOTAL   :", total_known)
print("CORRECT :", correct_known)
print("ACC     :", correct_known / total_known if total_known > 0 else 0)

print("\n--- UNKNOWN (10 persons) ---")
print("TOTAL   :", total_unknown)
print("CORRECT :", correct_unknown)
print("ACC     :", correct_unknown / total_unknown if total_unknown > 0 else 0)