"""
Face Registration State Machine
FRONT → LEFT → RIGHT → UP → DOWN → DONE
"""
import time, logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Dict, Callable
from core.head_pose import HeadPose, HeadPoseEstimator
import cv2, numpy as np

logger = logging.getLogger(__name__)


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

STEP_DIR = {
    Step.FRONT:"FRONT", Step.LEFT:"LEFT", Step.RIGHT:"RIGHT",
    Step.UP:"UP",       Step.DOWN:"DOWN",
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
    Step.FRONT:"⬜", Step.LEFT:"⬅️", Step.RIGHT:"➡️",
    Step.UP:"⬆️",   Step.DOWN:"⬇️", Step.DONE:"✅",
}


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


class FaceRegistrationSession:
    def __init__(self, user_id: str,
                 hold_seconds: float = 1.5,
                 frames_per_step: int = 5,
                 step_timeout: float = 30.0):
        self.user_id = user_id
        self.hold_seconds = hold_seconds
        self.frames_per_step = frames_per_step
        self.step_timeout = step_timeout
        self._estimator = HeadPoseEstimator()
        self._result = RegistrationResult(user_id=user_id)
        self._idx = 0
        self._reset_state()

    @property
    def current_step(self) -> Step: return STEP_ORDER[self._idx]

    def is_done(self) -> bool: return self.current_step == Step.DONE

    def _reset_state(self):
        self._hold_start: Optional[float] = None
        self._step_start = time.time()
        self._captured = 0
        self._last_prompt = 0.0

    def get_progress(self) -> Dict:
        total = len(STEP_ORDER) - 1
        done  = self._idx
        hold  = 0.0
        if self._hold_start:
            hold = min((time.time() - self._hold_start) / self.hold_seconds, 1.0)
        steps_status = []
        for i, s in enumerate(STEP_ORDER[:-1]):
            if i < done:   status = "done"
            elif i == done: status = "active"
            else:           status = "pending"
            steps_status.append({"step": s.name, "icon": STEP_ICON[s],
                                  "msg": STEP_MSG[s], "status": status})
        return {
            "step_index": done, "total": total,
            "percent": round(done / total * 100),
            "current_step": self.current_step.name,
            "instruction": STEP_MSG[self.current_step],
            "icon": STEP_ICON[self.current_step],
            "hold_progress": hold,
            "steps": steps_status,
        }

    def process_frame(self, frame_bgr: np.ndarray) -> Dict:
        if self.is_done():
            return self._ev("done", "Đăng ký hoàn tất!")

        pose = self._estimator.estimate(frame_bgr)
        prog = self.get_progress()

        if pose is None:
            return self._ev("no_face", "Không thấy khuôn mặt. Hãy đảm bảo đủ ánh sáng.", progress=prog)

        if time.time() - self._step_start > self.step_timeout:
            self._reset_step()
            return self._ev("timeout", f"Hết giờ! {STEP_MSG[self.current_step]}", pose=pose, progress=self.get_progress())

        req = STEP_DIR[self.current_step]
        if not DIRECTION_ANGLES[req].matches(pose):
            self._hold_start = None
            ev, msg = "waiting", STEP_MSG[self.current_step]
            if time.time() - self._last_prompt >= 3.0:
                self._last_prompt = time.time()
                msg = self._hint(pose, req)
                ev  = "prompt"
            return self._ev(ev, msg, pose=pose, progress=prog)

        # Đúng hướng
        if self._hold_start is None:
            self._hold_start = time.time()
        hold = min((time.time() - self._hold_start) / self.hold_seconds, 1.0)
        prog["hold_progress"] = hold

        if hold < 1.0:
            return self._ev("hold", f"Giữ nguyên... {int(hold*100)}%", pose=pose, progress=prog)

        # Capture
        cap = self._result.captures.setdefault(req, StepCapture(self.current_step, req))
        cap.frames.append(frame_bgr.copy())
        cap.poses.append(pose)
        self._captured += 1

        if self._captured >= self.frames_per_step:
            finished_step = self.current_step
            finished_dir = STEP_DIR[finished_step]
            self._idx += 1
            self._reset_state()
            if self.is_done():
                self._result.completed = True
                return self._ev("done", "✅ Đăng ký khuôn mặt hoàn tất!", pose=pose, progress=self.get_progress())
            ev = self._ev("step_done", f"✓ Xong! Tiếp theo: {STEP_MSG[self.current_step]}",
                          pose=pose, progress=self.get_progress())
            ev["finished_step"] = finished_step.name
            ev["finished_direction"] = finished_dir
            return ev

        return self._ev("captured", f"Đang chụp {self._captured}/{self.frames_per_step}",
                        pose=pose, progress=prog)

    def get_result(self): return self._result

    def _reset_step(self):
        req = STEP_DIR.get(self.current_step)
        if req and req in self._result.captures:
            del self._result.captures[req]
        self._reset_state()

    def redo_direction(self, direction: str):
        """
        Xoá capture của direction và quay lại bước đó để đăng ký lại.
        """
        try:
            direction = str(direction).upper()
        except Exception:
            return

        # xóa dữ liệu đã capture cho direction
        if direction in self._result.captures:
            del self._result.captures[direction]

        # set current step index về đúng step tương ứng
        for i, s in enumerate(STEP_ORDER[:-1]):
            if STEP_DIR.get(s) == direction:
                self._idx = i
                break
        self._reset_state()

    @staticmethod
    def _hint(pose: HeadPose, req: str) -> str:
        hints = {
            "LEFT":  f"Quay mặt sang TRÁI hơn nữa (Yaw: {pose.yaw:+.1f}°)",
            "RIGHT": f"Quay mặt sang PHẢI hơn nữa (Yaw: {pose.yaw:+.1f}°)",
            "UP":    f"Ngẩng đầu lên cao hơn (Pitch: {pose.pitch:+.1f}°)",
            "DOWN":  f"Cúi đầu xuống thấp hơn (Pitch: {pose.pitch:+.1f}°)",
            "FRONT": f"Nhìn thẳng vào camera",
        }
        return hints.get(req, STEP_MSG.get(Step[req], ""))

    @staticmethod
    def _ev(event, message, pose=None, progress=None):
        return {
            "event": event, "message": message,
            "pose": {"yaw": round(pose.yaw,1), "pitch": round(pose.pitch,1),
                     "roll": round(pose.roll,1), "direction": pose.direction()} if pose else None,
            "progress": progress,
        }

    def close(self): self._estimator.close()
