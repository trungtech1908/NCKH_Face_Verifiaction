"""
Qdrant Cloud Storage
--------------------
Mỗi user lưu 25 points (5 hướng × 5 frames).
Mỗi point = 1 embedding từ 1 frame, payload chỉ chứa user_id.

Khi search: query 1 embedding → top-k → trả user_id → caller query MySQL.
"""

import logging
import numpy as np
from typing import Optional, List, Tuple, Dict
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance,
    PointStruct, Filter, FieldCondition, MatchValue,
    PayloadSchemaType,
)
import config

logger = logging.getLogger(__name__)
EMBEDDING_DIM = 512


def _make_point_id(user_id: int, direction: str, frame_idx: int) -> int:
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

        # Payload index cho user_id
        try:
            self.client.create_payload_index(
                collection_name=config.QDRANT_COLLECTION,
                field_name="user_id",
                field_schema=PayloadSchemaType.INTEGER,
            )
        except Exception:
            pass

    # ── WRITE ──────────────────────────────────────────────────────────────

    def save_user_embeddings(
        self,
        user_id: int,
        captures: Dict,               # direction → StepCapture
        embedder,                     # FaceEmbedder instance
    ) -> int:
        """
        Extract embedding từng frame và lưu thành nhiều points.
        Payload chỉ chứa user_id (MySQL ID).
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
                        "user_id": user_id,
                    },
                ))

        if not points:
            return 0

        self.client.upsert(
            collection_name=config.QDRANT_COLLECTION,
            points=points,
        )
        logger.info("Saved %d points for user_id=%s", len(points), user_id)
        return len(points)

    def delete_user_embeddings(self, user_id: int):
        """Xóa tất cả embeddings của user_id."""
        self.client.delete(
            collection_name=config.QDRANT_COLLECTION,
            points_selector=Filter(
                must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
            ),
        )
        logger.info("Deleted embeddings for user_id=%s", user_id)

    def user_has_embeddings(self, user_id: int) -> bool:
        """Kiểm tra user đã có embeddings chưa."""
        results = self.client.scroll(
            collection_name=config.QDRANT_COLLECTION,
            scroll_filter=Filter(
                must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
            ),
            limit=1,
            with_payload=False,
            with_vectors=False,
        )
        return len(results[0]) > 0

    # ── FACE SEARCH ────────────────────────────────────────────────────────

    def query_face_points(
        self,
        embedding: np.ndarray,
        top_k: int = 15,
    ):
        """Trả về danh sách hits thô từ Qdrant."""
        try:
            hits = self.client.search(
                collection_name=config.QDRANT_COLLECTION,
                query_vector=embedding.tolist(),
                limit=top_k,
                with_payload=True,
            )
            return hits or []
        except AttributeError:
            results = self.client.query_points(
                collection_name=config.QDRANT_COLLECTION,
                query=embedding.tolist(),
                limit=top_k,
            )
            return getattr(results, "points", []) or []

    def has_any_face_match(
        self,
        embedding: np.ndarray,
        top_k: int = 15,
        threshold: float = None,
    ) -> Optional[Tuple[int, float]]:
        """
        Check khuôn mặt đã tồn tại trong DB.
        Trả (user_id, best_score) hoặc None.
        """
        if threshold is None:
            threshold = config.FACE_SIMILARITY_THRESHOLD

        hits = self.query_face_points(embedding=embedding, top_k=top_k)
        if not hits:
            return None

        good = [h for h in hits if getattr(h, "score", 0.0) >= threshold]
        if not good:
            return None

        best = max(good, key=lambda h: getattr(h, "score", 0.0))
        uid = best.payload.get("user_id")
        return uid, float(best.score)

    def match_face_like_demo_detailed(
        self,
        embedding: np.ndarray,
        top_k: int = 5,
        score_threshold: float = 0.6,
    ) -> Tuple[Optional[int], float]:
        """
        Cùng logic vote với match_face_like_demo.
        Trả (user_id hoặc None, điểm cosine hiển thị).
        Khi khớp: max score trong các hit >= ngưỡng thuộc user thắng.
        Khi không khớp: max score trong toàn bộ top hits (Unknown / chưa đủ vote).
        """
        from collections import Counter

        hits = self.query_face_points(embedding=embedding, top_k=top_k)
        if not hits:
            return None, 0.0

        max_any = max(float(getattr(h, "score", 0.0)) for h in hits)

        uids: List[int] = []
        uid_to_scores: Dict[int, List[float]] = {}
        for h in hits:
            sc = float(getattr(h, "score", 0.0))
            if sc < score_threshold:
                continue
            uid = h.payload.get("user_id")
            if uid is None:
                continue
            try:
                ui = int(uid)
            except (TypeError, ValueError):
                continue
            uids.append(ui)
            uid_to_scores.setdefault(ui, []).append(sc)

        if not uids:
            return None, round(max_any, 4)

        best_uid, cnt = Counter(uids).most_common(1)[0]
        if cnt < 2:
            return None, round(max_any, 4)

        winner_scores = uid_to_scores.get(best_uid, [])
        rep = max(winner_scores) if winner_scores else max_any
        return best_uid, round(float(rep), 4)

    def match_face_like_demo(
        self,
        embedding: np.ndarray,
        top_k: int = 5,
        score_threshold: float = 0.6,
    ) -> Optional[int]:
        """
        Giống test_va_crop_face: query top_k, lọc score >= ngưỡng,
        majority vote theo user_id, cần ít nhất 2 hit cùng user_id.
        """
        uid, _ = self.match_face_like_demo_detailed(
            embedding, top_k=top_k, score_threshold=score_threshold
        )
        return uid

    def search_by_face(
        self,
        embedding: np.ndarray,
        top_k: int = 10,
        threshold: float = None,
    ) -> Optional[Tuple[int, float]]:
        """
        Tìm user bằng khuôn mặt (majority vote).
        Trả (user_id, avg_score) hoặc None.
        """
        if threshold is None:
            threshold = config.FACE_SIMILARITY_THRESHOLD

        hits = self.query_face_points(embedding=embedding, top_k=top_k)
        if not hits:
            return None

        # Filter by threshold
        hits = [h for h in hits if getattr(h, "score", 0.0) >= threshold]
        if not hits:
            return None

        # Majority vote theo user_id
        from collections import defaultdict
        vote = defaultdict(list)
        for h in hits:
            uid = h.payload.get("user_id")
            if uid is not None:
                vote[uid].append(h.score)

        if not vote:
            return None

        winner = max(vote, key=lambda uid: sum(vote[uid]) / len(vote[uid]))
        avg_score = sum(vote[winner]) / len(vote[winner])
        logger.info("Face search winner=%s votes=%d avg=%.3f",
                    winner, len(vote[winner]), avg_score)
        return winner, avg_score
