import os
import cv2
import uuid
from dotenv import load_dotenv
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

total_images = 0
pushed = 0

persons = os.listdir(DATASET_PATH)

for person in tqdm(persons, desc="Persons"):

    person_path = os.path.join(DATASET_PATH, person)
    if not os.path.isdir(person_path):
        continue

    images = sorted(os.listdir(person_path))
    total_images += len(images)

    gallery_images = images[20:]

    for img_name in tqdm(gallery_images, desc=f"{person}", leave=False):

        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        face = cv2.resize(img, (112, 112))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        emb = rec_model.get_feat(face).flatten()

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=[{
                "id": str(uuid.uuid4()),
                "vector": emb.tolist(),
                "payload": {"username": person}
            }]
        )

        pushed += 1

print("TOTAL IMAGES:", total_images)
print("TOTAL PUSHED:", pushed)