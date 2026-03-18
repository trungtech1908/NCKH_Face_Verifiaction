"""
Head Pose Estimation — MediaPipe 0.10+ Tasks API
Dùng ratio-based method để tránh gimbal lock của solvePnP.
"""
import os, urllib.request, cv2, numpy as np, mediapipe as mp, logging
from dataclasses import dataclass
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")

def _ensure_model():
    if not os.path.exists(MODEL_PATH):
        logger.info("Downloading face_landmarker.task (~30MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

_NOSE_TIP     = 1
_CHIN         = 152
_LEFT_CHEEK   = 234
_RIGHT_CHEEK  = 454
_FOREHEAD     = 10
_LEFT_EYE_OUT = 263
_RIGHT_EYE_OUT= 33


@dataclass
class HeadPose:
    yaw: float    # âm=trái, dương=phải
    pitch: float  # dương=lên, âm=xuống
    roll: float

    def direction(self) -> str:
        if abs(self.yaw) <= 18 and abs(self.pitch) <= 18:
            return "FRONT"
        if abs(self.yaw) >= abs(self.pitch):
            return "LEFT" if self.yaw < -18 else "RIGHT"
        return "UP" if self.pitch > 18 else "DOWN"

    def __str__(self):
        return f"Yaw={self.yaw:+.1f} Pitch={self.pitch:+.1f} [{self.direction()}]"


class HeadPoseEstimator:
    def __init__(self, confidence: float = 0.5):
        _ensure_model()
        opts = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=confidence,
            min_face_presence_confidence=confidence,
            min_tracking_confidence=confidence,
        )
        self._lm = mp.tasks.vision.FaceLandmarker.create_from_options(opts)

    def estimate(self, frame_bgr: np.ndarray) -> Optional[HeadPose]:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self._lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if not res.face_landmarks:
            return None
        lm = res.face_landmarks[0]
        def pt(i): return np.array([lm[i].x * w, lm[i].y * h])

        nose, lch, rch = pt(_NOSE_TIP), pt(_LEFT_CHEEK), pt(_RIGHT_CHEEK)
        d_left  = nose[0] - lch[0]
        d_right = rch[0] - nose[0]
        total   = d_left + d_right
        if total < 1e-6: return None
        yaw = (d_left / total - 0.5) * 180.0

        fh, chin = pt(_FOREHEAD), pt(_CHIN)
        face_h = chin[1] - fh[1]
        if face_h < 1e-6: return None
        pitch = -(( nose[1] - fh[1]) / face_h - 0.55) * 200.0

        le, re = pt(_LEFT_EYE_OUT), pt(_RIGHT_EYE_OUT)
        roll = float(np.degrees(np.arctan2(re[1]-le[1], re[0]-le[0])))

        return HeadPose(yaw=float(yaw), pitch=float(pitch), roll=roll)

    def close(self): self._lm.close()
    def __enter__(self): return self
    def __exit__(self, *_): self.close()
