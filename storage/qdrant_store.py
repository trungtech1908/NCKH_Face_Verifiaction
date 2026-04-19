"""
Qdrant Cloud Storage
--------------------
Mỗi user lưu 25 points (5 hướng × 5 frames).
Mỗi point = 1 embedding từ 1 frame, kèm payload user info.

Khi search (login/verify): query 1 embedding → top-k → majority vote theo user_id.
"""

import logging
import numpy as np
from typing import Optional, List, Tuple, Dict
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance,
    PointStruct, Filter, FieldCondition, MatchValue,
    UpdateStatus, PayloadSchemaType,
)
import config

logger = logging.getLogger(__name__)
EMBEDDING_DIM = 512


def _make_point_id(user_id: str, direction: str, frame_idx: int) -> int:
    """Tạo unique int id cho mỗi point: hash(user_id + direction + idx)."""
    import hashlib
    key = f"{user_id}:{direction}:{frame_idx}"
    h = hashlib.md5(key.encode()).hexdigest()
    return int(h[:16], 16) % (2**63)


class QdrantFaceStore:
    def __init__(self):
        self.client = QdrantClient(
            url=config.QDRANT_URL,
            api_key=config.QDRANT_API_KEY,
            timeout=15,
        )
        self._ensure_collection()

    def _ensure_collection(self):
        existing = [c.name for c in self.client.get_collections().collections]
        if config.QDRANT_COLLECTION not in existing:
            self.client.create_collection(
                collection_name=config.QDRANT_COLLECTION,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
            )
            logger.info("Created collection '%s'", config.QDRANT_COLLECTION)
        else:
            logger.info("Collection '%s' ready", config.QDRANT_COLLECTION)

        # Payload indexes bắt buộc cho Qdrant Cloud
        for field in ("username", "email", "user_id"):
            try:
                self.client.create_payload_index(
                    collection_name=config.QDRANT_COLLECTION,
                    field_name=field,
                    field_schema=PayloadSchemaType.KEYWORD,
                )
            except Exception:
                pass  # index đã tồn tại

    # ── WRITE ──────────────────────────────────────────────────────────────

    def save_user_embeddings(
        self,
        user_id: str,
        user_payload: Dict,           # username, email, password, created_at
        captures: Dict,               # direction → StepCapture
        embedder,                     # FaceEmbedder instance
    ) -> int:
        """
        Extract embedding từng frame và lưu thành nhiều points.
        Trả về số points đã lưu.
        """
        points = []
        for direction, capture in captures.items():
            for idx, frame in enumerate(capture.frames):
                emb = embedder.extract(frame)
                if emb is None:
                    logger.warning("Skip frame %s[%d]: no face", direction, idx)
                    continue

                point_id = _make_point_id(user_id, direction, idx)
                points.append(PointStruct(
                    id=point_id,
                    vector=emb.tolist(),
                    payload={
                        "user_id":   user_id,
                        "direction": direction,
                        "frame_idx": idx,
                        **user_payload,
                    },
                ))

        if not points:
            return 0

        self.client.upsert(
            collection_name=config.QDRANT_COLLECTION,
            points=points,
        )
        logger.info("Saved %d points for user=%s", len(points), user_id)
        return len(points)

    def delete_user(self, user_id: str):
        self.client.delete(
            collection_name=config.QDRANT_COLLECTION,
            points_selector=Filter(
                must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
            ),
        )

    # ── READ ───────────────────────────────────────────────────────────────

    def get_user_by_field(self, field: str, value: str) -> Optional[Dict]:
        results = self.client.scroll(
            collection_name=config.QDRANT_COLLECTION,
            scroll_filter=Filter(
                must=[FieldCondition(key=field, match=MatchValue(value=value))]
            ),
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        points = results[0]
        return points[0].payload if points else None

    def get_user_by_username(self, username: str) -> Optional[Dict]:
        return self.get_user_by_field("username", username)

    def get_user_by_email(self, email: str) -> Optional[Dict]:
        return self.get_user_by_field("email", email)

    def get_user_by_id(self, user_id: str) -> Optional[Dict]:
        return self.get_user_by_field("user_id", user_id)

    def user_exists(self, user_id: str) -> bool:
        return self.get_user_by_id(user_id) is not None

    # ── FACE SEARCH ────────────────────────────────────────────────────────

    def search_by_face(
        self,
        embedding: np.ndarray,
        top_k: int = 10,
        threshold: float = None,
    ) -> Optional[Tuple[Dict, float]]:
        """
        Tìm user bằng khuôn mặt:
          1. Vector search → top_k points gần nhất
          2. Majority vote theo user_id
          3. Trả về (user_payload, avg_score) của winner, hoặc None
        """
        if threshold is None:
            threshold = config.FACE_SIMILARITY_THRESHOLD

        # qdrant-client version compatibility:
        # - Newer versions support `search(...)`
        # - Some versions (incl. used in test.py) use `query_points(...)`
        hits = None
        try:
            hits = self.client.search(
                collection_name=config.QDRANT_COLLECTION,
                query_vector=embedding.tolist(),
                limit=top_k,
                score_threshold=threshold,
                with_payload=True,
            )
        except AttributeError:
            results = self.client.query_points(
                collection_name=config.QDRANT_COLLECTION,
                query=embedding.tolist(),
                limit=top_k,
            )
            hits = getattr(results, "points", None)

        if not hits:
            return None

        # Enforce threshold for clients that don't support score_threshold in query_points
        try:
            hits = [h for h in hits if getattr(h, "score", 0.0) >= threshold]
        except Exception:
            pass
        if not hits:
            return None

        # Tổng hợp score theo user_id
        vote: Dict[str, List[float]] = {}
        payloads: Dict[str, Dict] = {}
        for h in hits:
            uid = h.payload.get("user_id")
            if uid:
                vote.setdefault(uid, []).append(h.score)
                payloads[uid] = h.payload

        if not vote:
            return None

        # Winner = user có avg score cao nhất
        winner = max(vote, key=lambda uid: sum(vote[uid]) / len(vote[uid]))
        avg_score = sum(vote[winner]) / len(vote[winner])
        logger.info("Face search winner=%s votes=%d avg=%.3f",
                    winner, len(vote[winner]), avg_score)
        return payloads[winner], avg_score
