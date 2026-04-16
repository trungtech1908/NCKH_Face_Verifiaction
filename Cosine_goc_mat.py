import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import os

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

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

# ================== FILTER USERNAME ==================
qfilter = Filter(
    must=[
        FieldCondition(
            key="username",
            match=MatchValue(value="trung")
        )
    ]
)

# ================== FETCH VECTOR ==================
res = client.scroll(
    collection_name=COLLECTION_NAME,
    scroll_filter=qfilter,
    limit=100,
    with_vectors=True,
    with_payload=True
)

points = res[0]

vectors = []
labels = []

for p in points:
    if p.vector is not None:
        vectors.append(p.vector)
        labels.append(str(p.id))

vectors = np.array(vectors)

# ================== COSINE SIMILARITY ==================
sim_matrix = cosine_similarity(vectors)

# ================== HEATMAP ==================
plt.figure(figsize=(10, 8))
plt.imshow(sim_matrix, cmap="hot")
plt.colorbar()

plt.xticks(range(len(labels)), labels, rotation=90, fontsize=6)
plt.yticks(range(len(labels)), labels, fontsize=6)

plt.title("Cosine Heatmap - username = trung")
plt.tight_layout()
plt.show()