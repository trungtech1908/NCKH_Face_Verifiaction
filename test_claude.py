import os
import cv2
from dotenv import load_dotenv
from collections import Counter
from insightface.app import FaceAnalysis
from qdrant_client import QdrantClient
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

load_dotenv()
client = QdrantClient(url=os.getenv("URL_QDRANT"), api_key=os.getenv("API_QDRANT"), timeout=60 )
COLLECTION_NAME = "NCKH_Face_Verification"

app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0)
rec_model = app.models['recognition']

DATASET_PATH = "megaface-testsuite/megaface/data/megaface_testpack_v1.0/facescrub_images"
TOP_K = 15
THRESHOLD = 0.4
VOTE_K = 1

total = 0
correct = 0
cm = np.zeros((2, 2), dtype=int)  # [actual][pred]: 0=InDB, 1=Unknown

persons = sorted(os.listdir(DATASET_PATH))
persons = [p for p in persons if os.path.isdir(os.path.join(DATASET_PATH, p))]
gallery_persons = set(persons[:70])

for person in tqdm(persons, desc="Persons"):
    person_path = os.path.join(DATASET_PATH, person)
    images = sorted(os.listdir(person_path))
    in_db = person in gallery_persons
    test_images = images[20:] if in_db else images

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

        if in_db:
            actual_idx = 0
            if pred == person:       # TP
                pred_idx = 0
                correct += 1
            else:
                pred_idx = 1         # FN (predict Unknown hoặc sai tên)
        else:
            actual_idx = 1
            if pred == "Unknown":    # TN
                pred_idx = 1
                correct += 1
            else:
                pred_idx = 0         # FP

        cm[actual_idx][pred_idx] += 1

# ===== METRICS =====
TP, FN = cm[0][0], cm[0][1]
FP, TN = cm[1][0], cm[1][1]

precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall    = TP / (TP + FN) if (TP + FN) > 0 else 0
f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
far       = FP / (FP + TN) if (FP + TN) > 0 else 0
frr       = FN / (FN + TP) if (FN + TP) > 0 else 0

print(f"\n{'='*40}")
print(f"TOTAL      : {total}")
print(f"CORRECT    : {correct}  (TP={TP}, TN={TN})")
print(f"WRONG      : {total - correct}  (FP={FP}, FN={FN})")
print(f"ACC        : {correct / total:.4f}")
print(f"Precision  : {precision:.4f}")
print(f"Recall     : {recall:.4f}")
print(f"F1         : {f1:.4f}")
print(f"FAR        : {far:.4f}  (False Accept Rate)")
print(f"FRR        : {frr:.4f}  (False Reject Rate)")
print(f"{'='*40}\n")

# ===== CONFUSION MATRIX =====
labels = ["In DB", "Unknown"]
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(labels, fontsize=11)
ax.set_yticklabels(labels, fontsize=11)
ax.set_xlabel("Predicted", fontsize=12)
ax.set_ylabel("Actual", fontsize=12)
ax.set_title(f"Confusion Matrix  |  ACC = {correct / total:.4f}", fontsize=13)

cell_labels = [["TP", "FN"], ["FP", "TN"]]
for i in range(2):
    for j in range(2):
        color = "white" if cm[i, j] > cm.max() / 2 else "black"
        ax.text(j, i, f"{cell_labels[i][j]}\n{cm[i, j]}",
                ha="center", va="center",
                color=color, fontsize=13, fontweight="bold")

plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()
print("Saved: confusion_matrix.png")