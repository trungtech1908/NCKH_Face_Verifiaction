"""
Face Embedding — InsightFace ArcFace 512-d
"""
import logging
import numpy as np
from typing import Optional, List

logger = logging.getLogger(__name__)


class FaceEmbedder:
    """InsightFace buffalo_l → ArcFace 512-d embedding."""

    def __init__(self, det_size=(320, 320)):
        from insightface.app import FaceAnalysis
        self.app = FaceAnalysis(name="buffalo_l",
                                providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.app.prepare(ctx_id=0, det_size=det_size)
        logger.info("InsightFace buffalo_l loaded")

    def extract(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        faces = self.app.get(frame_bgr)
        if not faces:
            return None
        face = max(faces, key=lambda f: f.det_score)
        return face.embedding

    def extract_best(self, frames: List[np.ndarray]) -> Optional[np.ndarray]:
        """Lấy embedding tốt nhất từ list frames (det_score cao nhất)."""
        best_emb, best_score = None, -1.0
        for f in frames:
            faces = self.app.get(f)
            if not faces:
                continue
            face = max(faces, key=lambda x: x.det_score)
            if face.det_score > best_score:
                best_score = face.det_score
                best_emb   = face.embedding
        return best_emb


def aggregate_embeddings(embeddings: List[np.ndarray],
                         threshold: float = 0.4) -> Optional[np.ndarray]:
    """Mean embedding sau khi lọc outlier."""
    if not embeddings:
        return None
    if len(embeddings) == 1:
        return embeddings[0]
    stack = np.stack(embeddings)
    mean  = stack.mean(axis=0)
    sims  = stack @ mean
    good  = stack[sims >= threshold]
    if len(good) == 0:
        return mean
    return good.mean(axis=0)


def build_user_embedding(captures: dict,
                         embedder: FaceEmbedder) -> Optional[np.ndarray]:
    """
    Từ dict captures (direction → StepCapture),
    extract 1 embedding đại diện cho toàn bộ khuôn mặt.
    """
    per_dir = []
    for direction, cap in captures.items():
        emb = embedder.extract_best(cap.frames)
        if emb is not None:
            per_dir.append(emb)
            logger.info("Direction %s: embedding OK", direction)
        else:
            logger.warning("Direction %s: no face detected", direction)

    if not per_dir:
        return None

    final = aggregate_embeddings(per_dir)
    logger.info("Final embedding shape=%s", final.shape if final is not None else None)
    return final
