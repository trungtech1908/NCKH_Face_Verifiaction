import time
import logging
import torch
import cv2
import os
import numpy as np
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Dict, Callable
from collections import OrderedDict, deque

# Giả định các module này đã có sẵn trong source của bạn
# from core.head_pose import HeadPose, HeadPoseEstimator
# from src.model_lib.MiniFASNet import MiniFASNetV2, MiniFASNetV1SE
# from src.utility import get_crop_face

logger = logging.getLogger(__name__)

@dataclass
class AngleRange:
    yaw_min: float
    yaw_max: float
    pitch_min: float
    pitch_max: float
    
    def matches(self, p) -> bool:
        return self.yaw_min <= p.yaw <= self.yaw_max and self.pitch_min <= p.pitch <= self.pitch_max

DIRECTION_ANGLES = {
    "FRONT": AngleRange(-18, 18, -18, 18),
    "LEFT":  AngleRange(-90, -22, -25, 25),
    "RIGHT": AngleRange(22,  90,  -25, 25),
    "UP":    AngleRange(-25, 25,  18,  80),
    "DOWN":  AngleRange(-25, 25, -80, -18),
}

class Step(Enum):
    FRONT = auto()
    LEFT = auto()
    RIGHT = auto()
    UP = auto()
    DOWN = auto()
    DONE = auto()

STEP_ORDER = [Step.FRONT, Step.LEFT, Step.RIGHT, Step.UP, Step.DOWN, Step.DONE]

STEP_DIR = {
    Step.FRONT: "FRONT", Step.LEFT: "LEFT", Step.RIGHT: "RIGHT",
    Step.UP: "UP",       Step.DOWN: "DOWN",
}

STEP_MSG = {
    Step.FRONT: "Nhìn thẳng vào camera",
    Step.LEFT:  "Quay mặt sang TRÁI",
    Step.RIGHT: "Quay mặt sang PHẢI",
    Step.UP:    "Ngẩng mặt lên TRÊN",
    Step.DOWN:  "Cúi mặt xuống DƯỚI",
    Step.DONE:  "Đăng ký hoàn tất!",
}

STEP_ICON = {
    Step.FRONT: "⬜", Step.LEFT: "⬅️", Step.RIGHT: "➡️",
    Step.UP: "⬆️",   Step.DOWN: "⬇️", Step.DONE: "✅",
}

@dataclass
class StepCapture:
    step: Step
    direction: str
    frames: List = field(default_factory=list)
    poses:  List = field(default_factory=list)

@dataclass
class RegistrationResult:
    user_id: str
    captures: Dict[str, StepCapture] = field(default_factory=dict)
    completed: bool = False
    
    def frame_count(self): 
        return sum(len(c.frames) for c in self.captures.values())

class AntiSpoof:
    def __init__(self, model_dir):
        from src.model_lib.MiniFASNet import MiniFASNetV2, MiniFASNetV1SE # Import tại đây nếu cần
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir = model_dir
        self.model_27 = self._load_model("2.7_80x80_MiniFASNetV2.pth", MiniFASNetV2)
        self.model_4_0 = self._load_model("4_0_0_80x80_MiniFASNetV1SE.pth", MiniFASNetV1SE)
        print(">>> [SUCCESS] FAS Models loaded with RGB mode.")

    def _load_model(self, model_name, model_class):
        model_path = os.path.join(self.model_dir, model_name)
        model = model_class(conv6_kernel=5, num_classes=3)
        state_dict = torch.load(model_path, map_location=self.device)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        model.to(self.device)
        model.eval()
        return model

    def preprocessing(self, img):
        if img is None or img.size == 0: return None
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_res = cv2.resize(img_rgb, (80, 80))
        img_data = img_res.astype('float32') 
        img_data = img_data.transpose((2, 0, 1))
        return torch.tensor(img_data).to(self.device)

    def predict(self, img_27, img_40):
        t_27 = self.preprocessing(img_27)
        t_40 = self.preprocessing(img_40)
        if t_27 is None or t_40 is None: return 0, 0.0

        with torch.no_grad():
            out_27 = self.model_27(t_27.unsqueeze(0))
            out_40 = self.model_4_0(t_40.unsqueeze(0))
            p27 = torch.softmax(out_27, dim=1)
            p40 = torch.softmax(out_40, dim=1)
            combined = (p27 + p40) / 2
            score, label = torch.max(combined, 1)
            return label.item(), score.item()

class FaceRegistrationSession:
    def __init__(self, user_id: str, hold_seconds: float = 1.5, frames_per_step: int = 5,
                 step_timeout: float = 30.0, face_app=None):
        """face_app: an insightface.app.FaceAnalysis instance (recommended to reuse the same app).
        If not provided, a local FaceAnalysis will be created as a fallback.
        """
        from core.head_pose import HeadPoseEstimator
        from core.embedding import FaceEmbedder

        self.user_id = user_id
        self.hold_seconds = hold_seconds
        self.frames_per_step = frames_per_step
        self.step_timeout = step_timeout

        # InsightFace app (detection + recognition) — passed in to keep detection consistent
        if face_app is None:
            try:
                from insightface.app import FaceAnalysis
                face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
                face_app.prepare(ctx_id=0)
            except Exception:
                face_app = None
        self._face_app = face_app

        # Head pose estimator and embedder
        self._estimator = HeadPoseEstimator()
        self._embedder = FaceEmbedder()

        self._result = RegistrationResult(user_id=user_id)
        self._idx = 0
        self._fas = AntiSpoof(model_dir="./external/anti_spoofing")

        # Anti-spoof buffers
        import config
        # store tuples (label, score) to allow checking label + score consistency
        self._fas_scores = deque(maxlen=config.ANTI_SPOOF_CONSISTENT_FRAMES)
        self._fas_prev_crop = None
        self._fas_static_count = 0

        self.reference_embedding = None

        self._reset_state()

    @property
    def current_step(self) -> Step:
        return STEP_ORDER[self._idx]

    def is_done(self) -> bool:
        return self.current_step == Step.DONE

    def _reset_state(self):
        self._hold_start: Optional[float] = None
        self._step_start = time.time()
        self._captured = 0
        self._last_prompt = 0.0

    def process_frame(self, frame_bgr: np.ndarray) -> Dict:
        from src.utility import get_crop_face
        from core.embedding import compute_similarity
        import config

        if self.is_done():
            return self._ev("done", "Đăng ký hoàn tất!")

        prog = self.get_progress()

        # Use InsightFace detection to get consistent bbox/landmarks
        faces = None
        if self._face_app is not None:
            try:
                faces = self._face_app.get(frame_bgr)
            except Exception:
                faces = None

        if not faces:
            # Fallback: try head_pose detector (slower)
            pose = self._estimator.estimate(frame_bgr)
            if pose is None:
                return self._ev("no_face", "Không thấy khuôn mặt", progress=prog)
            face_bbox = pose.bbox
        else:
            # prefer the best-scored face from InsightFace
            face = max(faces, key=lambda f: getattr(f, 'det_score', 0.0))
            face_bbox = face.bbox

        if time.time() - self._step_start > self.step_timeout:
            self._reset_step()
            return self._ev("timeout", "Hết giờ", progress=self.get_progress())

        # For head-pose estimation, run estimator on cropped face (avoid re-detection over full image)
        try:
            crop_for_pose = get_crop_face(frame_bgr, box=face_bbox, scale=1.0)
            pose = self._estimator.estimate(crop_for_pose)
            # estimator returns bbox relative to crop; override to the original insightface bbox for consistency
            if pose is not None:
                pose.bbox = face_bbox
        except Exception:
            pose = None

        if pose is None:
            return self._ev("no_face", "Không thấy khuôn mặt", progress=prog)

        req = STEP_DIR[self.current_step]
        if not DIRECTION_ANGLES[req].matches(pose):
            self._hold_start = None
            return self._ev("waiting", self._hint(pose, req), pose=pose, progress=prog)

        # Anti spoof (strict + temporal + static checks)
        i27 = get_crop_face(frame_bgr, box=face_bbox, scale=2.7)
        i40 = get_crop_face(frame_bgr, box=face_bbox, scale=4.0)
        label, score = self._fas.predict(i27, i40)

        # Immediate spoof detection: any frame not meeting strict per-frame criterion is rejected
        STRICT_SCORE = 0.9
        if not (label == 1 and score > STRICT_SCORE):
            # Reset hold and give clear feedback
            self._hold_start = None
            return self._ev("spoof_detected", "Phát hiện ảnh giả/2D!", pose=pose, progress=prog)

        # store label+score for temporal consistency (only real frames reach here)
        self._fas_scores.append((label, score))

        # Static (printed photo) detection: compare current crop with previous
        try:
            if self._fas_prev_crop is not None:
                prev = cv2.resize(self._fas_prev_crop, (80, 80))
                cur = cv2.resize(i40, (80, 80))
                mean_diff = float(np.mean(np.abs(prev.astype('int32') - cur.astype('int32'))))
                if mean_diff < config.ANTI_SPOOF_STATIC_THRESHOLD:
                    self._fas_static_count += 1
                else:
                    self._fas_static_count = 0
            self._fas_prev_crop = i40.copy()
        except Exception:
            self._fas_static_count = 0
            self._fas_prev_crop = i40.copy()

        if self._fas_static_count >= config.ANTI_SPOOF_STATIC_FRAMES:
            self._hold_start = None
            return self._ev("spoof_detected", "Phát hiện gian lận: ảnh tĩnh giống nhau", pose=pose, progress=prog)

        # Require several consecutive frames where label == 1 (temporal)
        passes = sum(1 for l, s in self._fas_scores if l == 1)
        if passes < config.ANTI_SPOOF_CONSISTENT_FRAMES:
            # Not yet confident about liveness; do not start hold timer
            self._hold_start = None
            return self._ev("spoof_uncertain", f"Chờ xác minh liveness... ({passes}/{config.ANTI_SPOOF_CONSISTENT_FRAMES})", pose=pose, progress=prog)

        # Hold logic — current frame is strict real already
        if self._hold_start is None:
            self._hold_start = time.time()

        hold = min((time.time() - self._hold_start) / self.hold_seconds, 1.0)
        prog["hold_progress"] = hold

        if hold < 1.0:
            return self._ev("hold", f"Giữ nguyên... {int(hold*100)}%", pose=pose, progress=prog)

        # Capture frame
        cap = self._result.captures.setdefault(req, StepCapture(self.current_step, req))
        cap.frames.append(frame_bgr.copy())
        cap.poses.append(pose)
        self._captured += 1

        # Enough frames → verify
        if self._captured >= self.frames_per_step:
            emb = self._embedder.extract_best(cap.frames)
            if emb is None:
                self._reset_step()
                return self._ev("error", "Không nhận diện được khuôn mặt", progress=prog)

            if self.current_step == Step.FRONT:
                self.reference_embedding = emb
                logger.info("Reference embedding set")
            else:
                sim = compute_similarity(self.reference_embedding, emb)
                logger.info(f"Similarity with reference: {sim:.3f}")
                if sim < config.FACE_VERIFICATION_THRESHOLD:
                    self._reset_step()
                    return self._ev(
                        "face_mismatch",
                        f"Khuôn mặt không khớp (sim={sim:.2f}) - thử lại",
                        progress=prog
                    )

            self._idx += 1
            self._reset_state()
            if self.is_done():
                self._result.completed = True
                return self._ev("done", "✅ Đăng ký hoàn tất!", progress=self.get_progress())
            return self._ev("step_done", f"✓ Tiếp theo: {STEP_MSG[self.current_step]}", progress=self.get_progress())

        return self._ev("captured", f"{self._captured}/{self.frames_per_step}", pose=pose, progress=prog)

    def get_result(self): return self._result

    def _reset_step(self):
        req = STEP_DIR.get(self.current_step)
        if req and req in self._result.captures:
            del self._result.captures[req]
        # Clear anti-spoof buffers for the next attempt
        try:
            self._fas_scores.clear()
            self._fas_prev_crop = None
            self._fas_static_count = 0
        except Exception:
            pass
        self._reset_state()

    @staticmethod
    def _hint(pose, req: str) -> str:
        hints = {
            "LEFT":  f"Quay mặt sang TRÁI hơn nữa (Yaw: {pose.yaw:+.1f}°)",
            "RIGHT": f"Quay mặt sang PHẢI hơn nữa (Yaw: {pose.yaw:+.1f}°)",
            "UP":    f"Ngẩng đầu lên cao hơn (Pitch: {pose.pitch:+.1f}°)",
            "DOWN":  f"Cúi đầu xuống thấp hơn (Pitch: {pose.pitch:+.1f}°)",
            "FRONT": "Nhìn thẳng vào camera",
        }
        return hints.get(req, STEP_MSG.get(Step[req] if req in Step.__members__ else None, ""))

    @staticmethod
    def _ev(event, message, pose=None, progress=None):
        return {
            "event": event, "message": message,
            "pose": {"yaw": round(pose.yaw, 1), "pitch": round(pose.pitch, 1),
                     "roll": round(pose.roll, 1), "direction": pose.direction()} if pose else None,
            "progress": progress,
        }

    def close(self): 
        self._estimator.close()
    def get_progress(self) -> Dict:
        total = len(STEP_ORDER) - 1
        done  = self._idx

        hold  = 0.0
        if self._hold_start:
            hold = min((time.time() - self._hold_start) / self.hold_seconds, 1.0)

        steps_status = []
        for i, s in enumerate(STEP_ORDER[:-1]):
            if i < done:
                status = "done"
            elif i == done:
                status = "active"
            else:
                status = "pending"

            steps_status.append({
            "step": s.name,
            "icon": STEP_ICON[s],
            "msg": STEP_MSG[s],
            "status": status
            })

        return {
        "step_index": done,
        "total": total,
        "percent": round(done / total * 100),
        "current_step": self.current_step.name,
        "instruction": STEP_MSG[self.current_step],
        "icon": STEP_ICON[self.current_step],
        "hold_progress": hold,
        "steps": steps_status,
        }