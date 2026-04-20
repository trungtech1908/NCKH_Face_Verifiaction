import os
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from itertools import combinations
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

# ===== ENV (GIỮ NGUYÊN) =====
load_dotenv()
client = QdrantClient(
    url=os.getenv("URL_QDRANT"),
    api_key=os.getenv("API_QDRANT"),
    timeout=30
)

COLLECTION_NAME = "NCKH_Face_Verification"

# ===== DATASET =====
DATASET_PATH = "megaface-testsuite/megaface/data/megaface_testpack_v1.0/facescrub_images"

persons = [
    name for name in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, name))
]

# ===== LƯU KẾT QUẢ =====
names = []
mean_values = []

# ===== COSINE =====
def cosine(a, b):
    return np.dot(a, b)

# ===== LOOP QUA TỪNG NGƯỜI =====
for USERNAME in persons:

    # 1. LẤY POINT
    points, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key="username",
                    match=MatchValue(value=USERNAME)
                )
            ]
        ),
        with_vectors=True,
        limit=10000
    )

    # 2. LẤY EMBEDDING
    embeddings = [np.array(p.vector) for p in points]

    # bỏ nếu không đủ dữ liệu
    if len(embeddings) < 2:
        continue

    # normalize
    embeddings = [e / np.linalg.norm(e) for e in embeddings]

    # 3. TÍNH COSINE(a, b), a != b
    cos_values = []

    for a, b in combinations(embeddings, 2):
        cos_values.append(cosine(a, b))

    cos_values = np.array(cos_values)

    # 4. COSINE TRUNG BÌNH
    mean_cos = np.mean(cos_values)

    print(f"{USERNAME}: {mean_cos:.4f}")

    names.append(USERNAME)
    mean_values.append(mean_cos)

# ===== THỐNG KÊ TỔNG =====
if len(mean_values) > 0:
    print("\n===== GLOBAL STATS =====")
    print("Mean:", np.mean(mean_values))
    print("Min :", np.min(mean_values))
    print("Max :", np.max(mean_values))

# ===== 5. VẼ HISTOGRAM =====
plt.figure()
plt.hist(mean_values, bins=50)
plt.xlabel("Mean cosine per person")
plt.ylabel("Frequency")
plt.title("Distribution of mean cosine (all persons)")
plt.show()