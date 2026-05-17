"""
Head Pose Estimation — dùng 5 landmark của InsightFace (RetinaFace).

Khác với phiên bản cũ (gọi MediaPipe Face Mesh để có 468 landmark), phiên bản
này tận dụng trực tiếp 5 keypoint mà InsightFace đã tính sẵn ở bước detect:

    kps[0] = mắt trái  (left eye, theo góc nhìn camera)
    kps[1] = mắt phải  (right eye)
    kps[2] = đầu mũi   (nose tip)
    kps[3] = góc miệng trái  (left mouth corner)
    kps[4] = góc miệng phải  (right mouth corner)

Cách tính 3 góc:

  • YAW (xoay quanh trục dọc — sang trái / phải):
        d_left  = x_nose - x_left_eye
        d_right = x_right_eye - x_nose
        yaw     = (d_left / (d_left + d_right) - 0.5) * 180

    Khi nhìn thẳng, mũi nằm giữa 2 mắt -> ratio ≈ 0.5 -> yaw ≈ 0.
    Quay mặt sang trái (camera-view): mũi dịch về phía mắt phải
    -> d_left tăng -> yaw > 0 (theo quy ước "+ = phải" của project hiện tại,
    file này giữ nguyên dấu cũ — xem `direction()`).

  • PITCH (xoay quanh trục ngang — ngẩng / cúi):
        InsightFace KHÔNG trả landmark trán/cằm, nên ta tổng hợp 2 mốc đó từ
        midpoint(eyes) và midpoint(mouth) bằng tỉ lệ giải phẫu (Farkas 1994):
            distance(eyes_mid → forehead) ≈ 0.5 × distance(eyes_mid → mouth_mid)
            distance(mouth_mid → chin)   ≈ 0.5 × distance(eyes_mid → mouth_mid)
        Sau đó dùng công thức ratio:
            face_h = chin_y - forehead_y
            pitch  = -((nose_y - forehead_y) / face_h - 0.50) * 200

        Lưu ý: offset 0.50 (không phải 0.55 như khi dùng landmark trán/cằm
        thật của MediaPipe) — vì với cặp trán/cằm tổng hợp, mũi nằm gần đúng
        giữa eye_mid và mouth_mid khi nhìn thẳng (theo Farkas 1994), nên
        baseline ratio ≈ 0.50, không phải 0.55. Dùng 0.55 sẽ làm baseline
        pitch ≈ +10° lúc nhìn thẳng → DOWN bị "trễ", UP "quá nhạy".

  • ROLL (nghiêng đầu):
        roll = atan2(y_right_eye - y_left_eye, x_right_eye - x_left_eye)
        (chính xác — không phụ thuộc anatomy.)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class HeadPose:
    yaw: float    # âm = trái, dương = phải
    pitch: float  # dương = ngẩng lên, âm = cúi xuống
    roll: float
    bbox: List[float]

    def direction(self) -> str:
        if abs(self.yaw) <= 18 and abs(self.pitch) <= 18:
            return "FRONT"
        if abs(self.yaw) >= abs(self.pitch):
            return "LEFT" if self.yaw < -18 else "RIGHT"
        return "UP" if self.pitch > 18 else "DOWN"

    def __str__(self) -> str:
        return f"Yaw={self.yaw:+.1f} Pitch={self.pitch:+.1f} [{self.direction()}]"


class HeadPoseEstimator:
    """
    Estimator nhận face object của InsightFace (có `.bbox` và `.kps`).

    Có thể truyền vào `estimate()`:
      • InsightFace face object (recommended — không phải detect lại).
      • numpy ndarray (frame BGR) — sẽ tự khởi tạo InsightFace lazily và detect.
    """

    def __init__(self, *_, face_app: Any = None, **__) -> None:
        # `face_app` chỉ dùng khi caller truyền ndarray vào `estimate()`. Nếu
        # caller luôn truyền face object có sẵn, attribute này có thể None.
        self._face_app = face_app

    @staticmethod
    def _from_landmarks(kps: np.ndarray, bbox: List[float]) -> Optional[HeadPose]:
        """Tính HeadPose từ 5 landmark (kps shape (5, 2)) + bbox."""
        if kps is None or len(kps) < 5:
            return None
        try:
            le = np.asarray(kps[0], dtype=np.float64)
            re = np.asarray(kps[1], dtype=np.float64)
            nose = np.asarray(kps[2], dtype=np.float64)
            lm = np.asarray(kps[3], dtype=np.float64)
            rm = np.asarray(kps[4], dtype=np.float64)
        except Exception:
            return None

        # ----- YAW: dùng 2 mắt + mũi -----
        d_left = nose[0] - le[0]
        d_right = re[0] - nose[0]
        total = d_left + d_right
        if abs(total) < 1e-6:
            return None
        yaw = (d_left / total - 0.5) * 180.0

        # ----- PITCH: tổng hợp trán + cằm từ midpoint(eyes) và midpoint(mouth) -----
        eye_mid = (le + re) * 0.5
        mouth_mid = (lm + rm) * 0.5
        em_vec = mouth_mid - eye_mid  # eye → mouth

        # Tỉ lệ giải phẫu Farkas: eye→forehead ≈ eye→mouth × 0.5,
        #                       mouth→chin   ≈ eye→mouth × 0.5
        forehead = eye_mid - 0.5 * em_vec
        chin = mouth_mid + 0.5 * em_vec

        face_h = chin[1] - forehead[1]
        if abs(face_h) < 1e-6:
            return None
        # Offset 0.50 (không phải 0.55 như khi dùng landmark trán/cằm thật của
        # MediaPipe): với cặp trán/cằm tổng hợp từ midpoint(eyes) và
        # midpoint(mouth), mũi nằm gần đúng giữa hai mốc khi nhìn thẳng
        # ⇒ baseline đối xứng giữa UP và DOWN, không lệch về phía UP.
        pitch = -((nose[1] - forehead[1]) / face_h - 0.50) * 200.0

        # ----- ROLL: từ 2 mắt -----
        roll = float(np.degrees(np.arctan2(re[1] - le[1], re[0] - le[0])))

        return HeadPose(
            yaw=float(yaw),
            pitch=float(pitch),
            roll=roll,
            bbox=[float(v) for v in bbox],
        )

    def estimate(self, face_or_frame: Any) -> Optional[HeadPose]:
        """
        - Nếu `face_or_frame` có thuộc tính `.kps` và `.bbox` (InsightFace face)
          ⇒ tính trực tiếp.
        - Nếu là numpy ndarray (frame BGR) ⇒ tự gọi InsightFace để detect rồi
          tính trên face có score cao nhất. Cần `face_app` được truyền vào
          constructor; nếu không có, tự khởi tạo lazily (nặng, không khuyến
          nghị).
        """
        # 1) face object đã có sẵn landmark
        if hasattr(face_or_frame, "kps") and hasattr(face_or_frame, "bbox"):
            face = face_or_frame
            kps = getattr(face, "kps", None)
            try:
                bbox = [float(v) for v in face.bbox]
            except Exception:
                return None
            return self._from_landmarks(np.asarray(kps), bbox)

        # 2) ndarray frame ⇒ detect rồi tính
        if isinstance(face_or_frame, np.ndarray):
            face_app = self._face_app
            if face_app is None:
                face_app = self._lazy_init_face_app()
            if face_app is None:
                return None
            try:
                faces = face_app.get(face_or_frame)
            except Exception:
                return None
            if not faces:
                return None
            face = max(faces, key=lambda f: float(getattr(f, "det_score", 0.0)))
            return self._from_landmarks(
                np.asarray(getattr(face, "kps", None)),
                [float(v) for v in face.bbox],
            )

        return None

    def _lazy_init_face_app(self) -> Any:
        try:
            from insightface.app import FaceAnalysis
        except Exception:
            logger.warning("InsightFace chưa được cài đặt — không thể fallback từ ndarray.")
            return None
        try:
            app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
            app.prepare(ctx_id=0)
        except Exception as e:
            logger.warning(f"Không khởi tạo được InsightFace: {e}")
            return None
        self._face_app = app
        return app

    def close(self) -> None:
        self._face_app = None

    def __enter__(self) -> "HeadPoseEstimator":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
