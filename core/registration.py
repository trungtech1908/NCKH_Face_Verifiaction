"""
Face Registration State Machine
FRONT → LEFT → RIGHT → UP → DOWN → DONE
Tích hợp Anti-Spoof (chống ảnh 2D / mặt nạ)
"""
import time, logging, os
import cv2
import numpy as np
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Dict
from core.head_pose import HeadPose, HeadPoseEstimator

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Anti-Spoof config mặc định (override bằng config.py nếu có)
# ─────────────────────────────────────────────
_DEFAULT_FAS_CFG = {
    "ANTI_SPOOF_SCORE_THRESHOLD":    0.6,
    "ANTI_SPOOF_CONSISTENT_FRAMES":  5,
    "ANTI_SPOOF_STATIC_THRESHOLD":   5.0,
    "ANTI_SPOOF_STATIC_FRAMES":      10,
}

def _fas_cfg(key: str):
    try:
        import config
        return getattr(config, key)
    except Exception:
        return _DEFAULT_FAS_CFG[key]


# ─────────────────────────────────────────────
# Anti-Spoof model wrapper
# ─────────────────────────────────────────────
class AntiSpoof:
    """Wrapper MiniFASNet V2 + V1SE → liveness score."""

    def __init__(self, model_dir: str = "./external/anti_spoofing"):
        import torch
        from src.model_lib.MiniFASNet import MiniFASNetV2, MiniFASNetV1SE
        self._torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir = model_dir
        self.model_27  = self._load("2.7_80x80_MiniFASNetV2.pth",       MiniFASNetV2)
        self.model_40  = self._load("4_0_0_80x80_MiniFASNetV1SE.pth",    MiniFASNetV1SE)
        logger.info("[AntiSpoof] Models loaded on %s", self.device)

    def _load(self, name, cls):
        torch = self._torch
        path = os.path.join(self.model_dir, name)
        model = cls(conv6_kernel=5, num_classes=3)
        sd = torch.load(path, map_location=self.device)
        clean = OrderedDict(
            (k[7:] if k.startswith("module.") else k, v) for k, v in sd.items()
        )
        model.load_state_dict(clean)
        model.to(self.device).eval()
        return model

    def _preprocess(self, img: np.ndarray):
        torch = self._torch
        if img is None or img.size == 0:
            return None
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (80, 80)).astype("float32")
        return torch.tensor(resized.transpose(2, 0, 1)).to(self.device)

    def predict(self, img_27: np.ndarray, img_40: np.ndarray):
        """Trả về (label, score). label=1 → real; label=0/2 → spoof."""
        torch = self._torch
        t27 = self._preprocess(img_27)
        t40 = self._preprocess(img_40)
        if t27 is None or t40 is None:
            return 0, 0.0
        with torch.no_grad():
            p27 = torch.softmax(self.model_27(t27.unsqueeze(0)), dim=1)
            p40 = torch.softmax(self.model_40(t40.unsqueeze(0)), dim=1)
            combined = (p27 + p40) / 2
            score, label = torch.max(combined, dim=1)
        return label.item(), score.item()


# ─────────────────────────────────────────────
# Angle / Step definitions
# ─────────────────────────────────────────────
@dataclass
class AngleRange:
    yaw_min: float; yaw_max: float
    pitch_min: float; pitch_max: float
    def matches(self, p: HeadPose) -> bool:
        return self.yaw_min <= p.yaw <= self.yaw_max and self.pitch_min <= p.pitch <= self.pitch_max

DIRECTION_ANGLES = {
    "FRONT": AngleRange(-18, 18, -18, 18),
    "LEFT":  AngleRange(-90, -22, -25, 25),
    "RIGHT": AngleRange(22,  90,  -25, 25),
    "UP":    AngleRange(-25, 25,  18,  80),
    "DOWN":  AngleRange(-25, 25, -80, -18),
}

class Step(Enum):
    FRONT = auto(); LEFT = auto(); RIGHT = auto()
    UP = auto(); DOWN = auto(); DONE = auto()

STEP_ORDER = [Step.FRONT, Step.LEFT, Step.RIGHT, Step.UP, Step.DOWN, Step.DONE]
STEP_DIR   = {Step.FRONT:"FRONT", Step.LEFT:"LEFT", Step.RIGHT:"RIGHT",
               Step.UP:"UP", Step.DOWN:"DOWN"}
STEP_MSG   = {
    Step.FRONT: "Nhìn thẳng vào camera",
    Step.LEFT:  "Quay mặt sang TRÁI",
    Step.RIGHT: "Quay mặt sang PHẢI",
    Step.UP:    "Ngẩng mặt lên TRÊN",
    Step.DOWN:  "Cúi mặt xuống DƯỚI",
    Step.DONE:  "Đăng ký hoàn tất!",
}
STEP_ICON  = {
    Step.FRONT:"⬜", Step.LEFT:"⬅️", Step.RIGHT:"➡️",
    Step.UP:"⬆️",   Step.DOWN:"⬇️", Step.DONE:"✅",
}


# ─────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────
@dataclass
class StepCapture:
    step: Step; direction: str
    frames: List = field(default_factory=list)
    poses:  List = field(default_factory=list)

@dataclass
class RegistrationResult:
    user_id: str
    captures: Dict[str, StepCapture] = field(default_factory=dict)
    completed: bool = False
    def frame_count(self): return sum(len(c.frames) for c in self.captures.values())


# ─────────────────────────────────────────────
# Main session
# ─────────────────────────────────────────────
class FaceRegistrationSession:
    def __init__(self, user_id: str,
                 hold_seconds: float = 1.5,
                 frames_per_step: int = 5,
                 step_timeout: float = 30.0,
                 fas_model_dir: str = "./external/anti_spoofing",
                 enable_anti_spoof: bool = False):
        self.user_id         = user_id
        self.hold_seconds    = hold_seconds
        self.frames_per_step = frames_per_step
        self.step_timeout    = step_timeout

        self._estimator = HeadPoseEstimator()
        self._result    = RegistrationResult(user_id=user_id)
        self._idx       = 0

        # Anti-spoof (tạm thời tắt theo yêu cầu)
        self._anti_spoof_enabled = bool(enable_anti_spoof)
        if self._anti_spoof_enabled:
            self._fas              = AntiSpoof(model_dir=fas_model_dir)
            self._fas_scores       = deque(maxlen=_fas_cfg("ANTI_SPOOF_CONSISTENT_FRAMES"))
            self._fas_prev_crop    = None
            self._fas_static_count = 0
        else:
            self._fas              = None
            self._fas_scores       = None
            self._fas_prev_crop    = None
            self._fas_static_count = 0

        self._reset_state()

    # ── properties ────────────────────────────
    @property
    def current_step(self) -> Step: return STEP_ORDER[self._idx]
    def is_done(self) -> bool:      return self.current_step == Step.DONE

    # ── state helpers ─────────────────────────
    def _reset_state(self):
        self._hold_start: Optional[float] = None
        self._step_start  = time.time()
        self._captured    = 0
        self._last_prompt = 0.0

    def _reset_step(self):
        req = STEP_DIR.get(self.current_step)
        if req and req in self._result.captures:
            del self._result.captures[req]
        # xoá buffer anti-spoof để thử lại sạch (nếu bật)
        if self._anti_spoof_enabled and self._fas_scores is not None:
            self._fas_scores.clear()
            self._fas_prev_crop    = None
            self._fas_static_count = 0
        self._reset_state()

    # ── progress ──────────────────────────────
    def get_progress(self) -> Dict:
        total = len(STEP_ORDER) - 1
        done  = self._idx
        hold  = 0.0
        if self._hold_start:
            hold = min((time.time() - self._hold_start) / self.hold_seconds, 1.0)
        steps_status = []
        for i, s in enumerate(STEP_ORDER[:-1]):
            if   i < done:   status = "done"
            elif i == done:  status = "active"
            else:            status = "pending"
            steps_status.append({"step": s.name, "icon": STEP_ICON[s],
                                  "msg": STEP_MSG[s], "status": status})
        return {
            "step_index":   done,
            "total":        total,
            "percent":      round(done / total * 100),
            "current_step": self.current_step.name,
            "instruction":  STEP_MSG[self.current_step],
            "icon":         STEP_ICON[self.current_step],
            "hold_progress": hold,
            "steps":        steps_status,
        }

    # ── main frame processor ──────────────────
    def process_frame(self, frame_bgr: np.ndarray) -> Dict:
        if self._anti_spoof_enabled:
            from src.utility import get_crop_face

        if self.is_done():
            return self._ev("done", "Đăng ký hoàn tất!")

        pose = self._estimator.estimate(frame_bgr)
        prog = self.get_progress()

        if pose is None:
            return self._ev("no_face", "Không thấy khuôn mặt. Hãy đảm bảo đủ ánh sáng.", progress=prog)

        if time.time() - self._step_start > self.step_timeout:
            self._reset_step()
            return self._ev("timeout", f"Hết giờ! {STEP_MSG[self.current_step]}",
                            pose=pose, progress=self.get_progress())

        req = STEP_DIR[self.current_step]

        # ── kiểm tra hướng mặt ────────────────
        if not DIRECTION_ANGLES[req].matches(pose):
            self._hold_start = None
            ev, msg = "waiting", STEP_MSG[self.current_step]
            if time.time() - self._last_prompt >= 3.0:
                self._last_prompt = time.time()
                msg = self._hint(pose, req)
                ev  = "prompt"
            return self._ev(ev, msg, pose=pose, progress=prog)

        # ── Anti-Spoof (tạm thời tắt) ─────────────────────────────
        # Khi bật lại, chỉ cần enable_anti_spoof=True ở constructor.
        if self._anti_spoof_enabled and self._fas is not None and self._fas_scores is not None:
            i27 = get_crop_face(frame_bgr, box=pose.bbox, scale=2.7)
            i40 = get_crop_face(frame_bgr, box=pose.bbox, scale=4.0)
            label, score = self._fas.predict(i27, i40)

            self._fas_scores.append(score)

            try:
                if self._fas_prev_crop is not None:
                    prev = cv2.resize(self._fas_prev_crop, (80, 80))
                    cur  = cv2.resize(i40,               (80, 80))
                    diff = float(np.mean(np.abs(prev.astype("int32") - cur.astype("int32"))))
                    if diff < _fas_cfg("ANTI_SPOOF_STATIC_THRESHOLD"):
                        self._fas_static_count += 1
                    else:
                        self._fas_static_count = 0
                self._fas_prev_crop = i40.copy()
            except Exception:
                self._fas_static_count = 0
                self._fas_prev_crop    = i40.copy() if i40 is not None else None

            if self._fas_static_count >= _fas_cfg("ANTI_SPOOF_STATIC_FRAMES"):
                self._hold_start = None
                return self._ev("spoof_detected",
                                "⚠️ Phát hiện gian lận: ảnh tĩnh / ảnh in 2D",
                                pose=pose, progress=prog)

            passes = sum(1 for s in self._fas_scores
                         if s >= _fas_cfg("ANTI_SPOOF_SCORE_THRESHOLD"))
            needed = _fas_cfg("ANTI_SPOOF_CONSISTENT_FRAMES")
            if passes < needed:
                self._hold_start = None
                return self._ev("spoof_uncertain",
                                f"Đang xác minh liveness... ({passes}/{needed})",
                                pose=pose, progress=prog)

        # ── Đúng hướng + liveness OK → hold timer ─
        if self._hold_start is None:
            self._hold_start = time.time()
        hold = min((time.time() - self._hold_start) / self.hold_seconds, 1.0)
        prog["hold_progress"] = hold

        if hold < 1.0:
            return self._ev("hold", f"Giữ nguyên... {int(hold*100)}%", pose=pose, progress=prog)

        # ── Capture frame ─────────────────────
        cap = self._result.captures.setdefault(req, StepCapture(self.current_step, req))
        cap.frames.append(frame_bgr.copy())
        cap.poses.append(pose)
        self._captured += 1

        if self._captured >= self.frames_per_step:
            finished_step = self.current_step
            finished_dir  = STEP_DIR[finished_step]
            self._idx    += 1
            self._reset_state()
            # reset anti-spoof buffer cho bước tiếp theo (nếu bật)
            if self._anti_spoof_enabled and self._fas_scores is not None:
                self._fas_scores.clear()
                self._fas_prev_crop    = None
                self._fas_static_count = 0

            if self.is_done():
                self._result.completed = True
                return self._ev("done", "✅ Đăng ký khuôn mặt hoàn tất!",
                                pose=pose, progress=self.get_progress())

            ev = self._ev("step_done",
                          f"✓ Xong! Tiếp theo: {STEP_MSG[self.current_step]}",
                          pose=pose, progress=self.get_progress())
            ev["finished_step"]      = finished_step.name
            ev["finished_direction"] = finished_dir
            return ev

        return self._ev("captured",
                        f"Đang chụp {self._captured}/{self.frames_per_step}",
                        pose=pose, progress=prog)

    # ── public helpers ────────────────────────
    def get_result(self): return self._result

    def redo_direction(self, direction: str):
        """Xoá capture của direction và quay lại bước đó để đăng ký lại."""
        try:
            direction = str(direction).upper()
        except Exception:
            return
        if direction in self._result.captures:
            del self._result.captures[direction]
        for i, s in enumerate(STEP_ORDER[:-1]):
            if STEP_DIR.get(s) == direction:
                self._idx = i
                break
        if self._anti_spoof_enabled and self._fas_scores is not None:
            self._fas_scores.clear()
            self._fas_prev_crop    = None
            self._fas_static_count = 0
        self._reset_state()

    def close(self): self._estimator.close()

    # ── static helpers ────────────────────────
    @staticmethod
    def _hint(pose: HeadPose, req: str) -> str:
        hints = {
            "LEFT":  f"Quay mặt sang TRÁI hơn nữa (Yaw: {pose.yaw:+.1f}°)",
            "RIGHT": f"Quay mặt sang PHẢI hơn nữa (Yaw: {pose.yaw:+.1f}°)",
            "UP":    f"Ngẩng đầu lên cao hơn (Pitch: {pose.pitch:+.1f}°)",
            "DOWN":  f"Cúi đầu xuống thấp hơn (Pitch: {pose.pitch:+.1f}°)",
            "FRONT": "Nhìn thẳng vào camera",
        }
        return hints.get(req, STEP_MSG.get(Step[req], ""))

    @staticmethod
    def _ev(event, message, pose=None, progress=None):
        return {
            "event":    event,
            "message":  message,
            "pose":     {"yaw":   round(pose.yaw, 1),
                         "pitch": round(pose.pitch, 1),
                         "roll":  round(pose.roll, 1),
                         "direction": pose.direction()} if pose else None,
            "progress": progress,
        }