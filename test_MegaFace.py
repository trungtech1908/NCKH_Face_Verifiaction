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
    api_key=os.getenv("API_QDRANT")
)

COLLECTION_NAME = "NCKH_Face_Verification"

# ===== MODEL =====
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0)
rec_model = app.models['recognition']

# ===== DATA =====
DATASET_PATH = "megaface-testsuite/megaface/data/megaface_testpack_v1.0/facescrub_images"

TOP_K = 15
THRESHOLD = 0.6
VOTE_K = 2

total = 0
correct = 0

persons = os.listdir(DATASET_PATH)

for person in tqdm(persons, desc="Persons"):

    person_path = os.path.join(DATASET_PATH, person)
    if not os.path.isdir(person_path):
        continue

    images = sorted(os.listdir(person_path))
    test_images = images[:21]

    for img_name in tqdm(test_images, desc=f"{person}", leave=False):

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

        total += 1
        if pred == person:
            correct += 1

print("TOTAL   :", total)
print("CORRECT :", correct)
print("ACC     :", correct / total if total > 0 else 0)