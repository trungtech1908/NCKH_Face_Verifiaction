"""
Phát hiện khuôn mặt bị che (khẩu trang / khăn / kính đen).

Chỉ dùng 2 tín hiệu heuristic dựa trên màu sắc, không dùng model riêng:
  - skin_ratio ở nửa dưới bbox (cằm/miệng): tụt mạnh khi đeo khẩu trang/khăn.
  - eye_dark_ratio quanh 2 keypoint mắt của InsightFace: tăng khi đeo kính đen.

Lưu ý: KHÔNG cố phát hiện tay che mặt (tay cùng tông da).
Trả về reason dạng code: "mask" / "glasses" để frontend tự dịch.
"""

from __future__ import annotations

from typing import Any, List, Tuple

import cv2
import numpy as np


MASK_SKIN_THR: float = 0.28
GLASSES_DARK_THR: float = 0.60
DARK_LEVEL: int = 50
MIN_FACE_SIDE: int = 40


def _skin_mask_ycrcb(bgr: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    _, cr, cb = cv2.split(ycrcb)
    mask = (
        (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)
    )
    return mask.astype(np.uint8)


def detect_occlusion(bgr: np.ndarray, face: Any) -> Tuple[bool, List[str], dict]:
    """
    Kiểm tra 1 khuôn mặt đã được InsightFace detect.
    `face` cần có `.bbox` (x1,y1,x2,y2) và, nếu có, `.kps` (5 landmarks).

    Returns:
        (occluded, reasons, metrics)
        reasons ⊂ {"mask", "glasses"}
    """
    h, w = bgr.shape[:2]
    try:
        x1, y1, x2, y2 = [int(float(v)) for v in face.bbox]
    except Exception:
        return False, [], {}

    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w - 1, x2))
    y2 = max(0, min(h - 1, y2))
    if (x2 - x1) < MIN_FACE_SIDE or (y2 - y1) < MIN_FACE_SIDE:
        return False, [], {}

    kps = getattr(face, "kps", None)

    if kps is not None and len(kps) >= 3:
        nose_y = int(kps[2][1])
    else:
        nose_y = y1 + int(0.55 * (y2 - y1))

    low_y1 = max(nose_y, y1 + int(0.45 * (y2 - y1)))
    low_y2 = y2
    if low_y2 - low_y1 < 8:
        return False, [], {}

    lower = bgr[low_y1:low_y2, x1:x2]
    if lower.size == 0:
        return False, [], {}

    skin = _skin_mask_ycrcb(lower)
    skin_ratio = float(skin.sum()) / float(skin.size)
    mask_detected = skin_ratio < MASK_SKIN_THR

    dark_ratios: List[float] = []
    if kps is not None and len(kps) >= 2:
        face_h = y2 - y1
        patch = max(12, int(0.18 * face_h))
        for i in range(2):
            cx, cy = int(kps[i][0]), int(kps[i][1])
            ex1 = max(0, cx - patch // 2)
            ey1 = max(0, cy - patch // 2)
            ex2 = min(w, cx + patch // 2)
            ey2 = min(h, cy + patch // 2)
            eye = bgr[ey1:ey2, ex1:ex2]
            if eye.size == 0:
                continue
            gray = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
            dark_ratios.append(float((gray < DARK_LEVEL).mean()))

    glasses_detected = (
        len(dark_ratios) == 2 and min(dark_ratios) > GLASSES_DARK_THR
    )

    reasons: List[str] = []
    if mask_detected:
        reasons.append("mask")
    if glasses_detected:
        reasons.append("glasses")

    metrics = {
        "skin_ratio": round(skin_ratio, 4),
        "dark_ratios": [round(r, 4) for r in dark_ratios],
    }
    return bool(reasons), reasons, metrics


def describe_reasons(reasons: List[str]) -> str:
    """Sinh chuỗi tiếng Việt từ list reason code."""
    if not reasons:
        return ""
    mapping = {
        "mask": "khẩu trang / khăn",
        "glasses": "kính đen",
    }
    parts = [mapping.get(r, r) for r in reasons]
    return "Khuôn mặt bị che (" + ", ".join(parts) + ")"
