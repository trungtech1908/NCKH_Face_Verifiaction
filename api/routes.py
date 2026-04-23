"""
FastAPI Routes
--------------
Account:
  POST /api/register          — đăng ký tài khoản (chỉ account, không face)
  POST /api/login             — đăng nhập, trả JWT kèm role

Face:
  POST /api/face/start/{uid}  — bắt đầu face registration session
  POST /api/face/{sid}/frame  — gửi frame
  POST /api/face/{sid}/finish — lưu embedding Qdrant + set face_registered
  POST /api/face/verify       — xác minh khuôn mặt

Admin:
  GET  /api/admin/users       — danh sách users
  POST /api/admin/users       — tạo user
  PUT  /api/admin/users/{id}  — sửa user
  DELETE /api/admin/users/{id} — xóa user + xóa embeddings

User:
  GET  /api/me                — thông tin user hiện tại

Pages:
  GET /                   — landing
  GET /register           — form đăng ký account
  GET /login              — form đăng nhập
  GET /admin              — admin dashboard
  GET /user               — user dashboard
  GET /face-register      — trang đăng ký khuôn mặt
"""

import threading, logging, time
import numpy as np
import cv2

from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional, Dict, List

from core.registration_FAS import FaceRegistrationSession
from core.embedding import FaceEmbedder, build_user_embedding, cosine_similarity
from core.occlusion import detect_occlusion, describe_reasons
from storage.qdrant_store import QdrantFaceStore
from storage.mysql_store import MySQLUserStore, ExamStore, init_database
from api.auth import (hash_password, verify_password,
                      create_access_token, decode_token)
import config

logger = logging.getLogger(__name__)

QDRANT_DUPLICATE_THRESHOLD = 0.5
SAME_PERSON_COSINE_THRESHOLD = 0.5

app = FastAPI(title="Face Verification System")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ── Singletons ─────────────────────────────────────────────────────────────
_qdrant: Optional[QdrantFaceStore] = None
_mysql: Optional[MySQLUserStore] = None
_exams: Optional[ExamStore] = None
_embedder: Optional[FaceEmbedder] = None
_embedder_lock = threading.Lock()
# Xác minh khuôn mặt: chỉ CPU (không dùng GPU cho InsightFace)
_verify_embedder: Optional[FaceEmbedder] = None
_verify_embedder_lock = threading.Lock()
_fas_antispoof = None
_fas_lock = threading.Lock()

# Giống test_va_crop_face: FAS real, Qdrant top-5 + ngưỡng + vote >= 2
VERIFY_FAS_LABEL_REAL = 1
VERIFY_FAS_SCORE_MIN = 0.9
VERIFY_QDRANT_TOP_K = 5
VERIFY_QDRANT_SCORE_MIN = 0.6
VERIFY_MAX_FACES_PER_FRAME = 8

# Cache kết quả theo embedding để skip FAS + Qdrant ở các frame kế tiếp (mặt đứng yên).
# Giống cách test_va_crop_face query Qdrant async — giúp FPS hiển thị sát với demo.
VERIFY_CACHE_TTL = 0.7                 # giây
VERIFY_CACHE_SIM = 0.85                # cosine ngưỡng — cùng một người
VERIFY_CACHE_MAX = 32
_verify_cache: list = []               # [(ts, emb_norm_np, payload_dict)]
_verify_cache_lock = threading.Lock()

def get_qdrant() -> QdrantFaceStore:
    global _qdrant
    if _qdrant is None:
        _qdrant = QdrantFaceStore()
    return _qdrant

def get_mysql() -> MySQLUserStore:
    global _mysql
    if _mysql is None:
        _mysql = MySQLUserStore()
    return _mysql

def get_exams() -> ExamStore:
    global _exams
    if _exams is None:
        _exams = ExamStore()
    return _exams

def get_embedder() -> FaceEmbedder:
    global _embedder
    if _embedder is None:
        with _embedder_lock:
            if _embedder is None:
                _embedder = FaceEmbedder()
    return _embedder


def get_verify_embedder() -> FaceEmbedder:
    """InsightFace chỉ CPU — dùng cho /api/face/verify."""
    global _verify_embedder
    if _verify_embedder is None:
        with _verify_embedder_lock:
            if _verify_embedder is None:
                _verify_embedder = FaceEmbedder(providers=["CPUExecutionProvider"])
    return _verify_embedder


def get_fas():
    """Anti-spoof (MiniFASNet) — dùng chung cho xác minh."""
    global _fas_antispoof
    if _fas_antispoof is None:
        with _fas_lock:
            if _fas_antispoof is None:
                from core.anti_spoof import AntiSpoof
                _fas_antispoof = AntiSpoof(model_dir="./external/anti_spoofing")
    return _fas_antispoof


def _infer_insightface_device(fa) -> str:
    try:
        models = getattr(fa, "models", None)
        if isinstance(models, dict) and models:
            for m in models.values():
                sess = getattr(m, "session", None)
                if sess is not None and hasattr(sess, "get_providers"):
                    providers = sess.get_providers()
                    return "GPU (CUDA)" if any("CUDA" in p for p in providers) else "CPU"
    except Exception:
        pass
    return "Unknown"


def _infer_fas_device(fas_model) -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "GPU (CUDA)"
        return "CPU"
    except Exception:
        return "Unknown"

# ── Init DB on startup ─────────────────────────────────────────────────────
@app.on_event("startup")
def startup():
    init_database()
    logger.info("MySQL initialized")

# ── Session store ───────────────────────────────────────────────────────────
_sessions: Dict[str, FaceRegistrationSession] = {}
_session_user: Dict[str, int] = {}  # sid → user_id (MySQL)
# Sau khi /finish thành công, session bị xóa; nếu client gọi /finish lặp lại thì trả lại kết quả cũ (200).
_face_finish_cache: Dict[str, dict] = {}
_lock = threading.Lock()

# ── Helpers ─────────────────────────────────────────────────────────────────
def _decode_frame(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Cannot decode image")
    return frame

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return cosine_similarity(a, b)

def _get_current_user(request: Request) -> dict:
    auth = request.headers.get("authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(401, "Missing token")
    token = auth.split(" ", 1)[1]
    payload = decode_token(token)
    if not payload:
        raise HTTPException(401, "Invalid or expired token")
    return payload

def _require_admin(request: Request) -> dict:
    user = _get_current_user(request)
    if user.get("role") != "admin":
        raise HTTPException(403, "Admin access required")
    return user

def _embedding_duplicate_check(qdrant: QdrantFaceStore, embedding: np.ndarray,
                                exclude_user_id: int = None) -> Optional[dict]:
    """Check khuôn mặt đã tồn tại trong Qdrant. Có thể exclude 1 user_id."""
    result = qdrant.has_any_face_match(
        embedding=embedding,
        top_k=15,
        threshold=QDRANT_DUPLICATE_THRESHOLD,
    )
    if result is None:
        return None
    uid, score = result
    if exclude_user_id is not None and uid == exclude_user_id:
        return None
    mysql = get_mysql()
    user = mysql.get_user_by_id(uid)
    return {"user_id": uid, "username": user.get("username", "Unknown") if user else "Unknown", "score": score}


# ══════════════════════════════════════════════════════════════════════════════
# PAGES
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/admin", response_class=HTMLResponse)
async def admin_page(request: Request):
    return templates.TemplateResponse("admin.html", {"request": request})

@app.get("/user", response_class=HTMLResponse)
async def user_page(request: Request):
    return templates.TemplateResponse("user_dashboard.html", {"request": request})

@app.get("/face-register", response_class=HTMLResponse)
async def face_register_page(request: Request):
    return templates.TemplateResponse("face_register.html", {"request": request})


# ══════════════════════════════════════════════════════════════════════════════
# API: REGISTER (chỉ account, không face)
# ══════════════════════════════════════════════════════════════════════════════

class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str
    full_name: str = ""
    student_id: str = ""

@app.post("/api/register")
async def register_account(body: RegisterRequest):
    mysql = get_mysql()
    if mysql.get_user_by_username(body.username):
        raise HTTPException(409, f"Username '{body.username}' đã tồn tại")
    if mysql.get_user_by_email(body.email):
        raise HTTPException(409, f"Email '{body.email}' đã được dùng")

    pwd_hash = hash_password(body.password)
    user_id = mysql.create_user(
        username=body.username,
        email=body.email,
        password_hash=pwd_hash,
        full_name=body.full_name,
        student_id=body.student_id,
    )
    if user_id is None:
        raise HTTPException(500, "Lỗi tạo tài khoản")

    return {"message": "Đăng ký thành công!", "user_id": user_id}


# ══════════════════════════════════════════════════════════════════════════════
# API: LOGIN
# ══════════════════════════════════════════════════════════════════════════════

class LoginRequest(BaseModel):
    username: str
    password: str

@app.post("/api/login")
async def login(body: LoginRequest):
    mysql = get_mysql()
    user = mysql.get_user_by_username(body.username)
    if not user:
        raise HTTPException(401, "Username không tồn tại")
    if not verify_password(body.password, user["password_hash"]):
        raise HTTPException(401, "Sai mật khẩu")

    token = create_access_token(user["id"], user["username"], user["role"])
    return {
        "message": "Đăng nhập thành công!",
        "token": token,
        "user": {
            "id": user["id"],
            "username": user["username"],
            "email": user["email"],
            "role": user["role"],
            "face_registered": user["face_registered"],
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# API: ME
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/me")
async def get_me(current: dict = Depends(_get_current_user)):
    mysql = get_mysql()
    user = mysql.get_user_by_id(current["sub"])
    if not user:
        raise HTTPException(404, "User không tồn tại")
    return {
        "id": user["id"],
        "username": user["username"],
        "email": user["email"],
        "full_name": user["full_name"],
        "student_id": user["student_id"],
        "role": user["role"],
        "face_registered": user["face_registered"],
        "created_at": user.get("created_at"),
    }


# ══════════════════════════════════════════════════════════════════════════════
# API: ADMIN — CRUD Users
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/admin/users")
async def admin_list_users(admin: dict = Depends(_require_admin)):
    """Chỉ trả về tài khoản role='user' (admin không thấy/không CRUD các admin khác)."""
    mysql = get_mysql()
    users = mysql.get_all_users()
    safe = []
    for u in users:
        if (u.get("role") or "").lower() == "admin":
            continue
        u.pop("password_hash", None)
        safe.append(u)
    return {"users": safe}

class AdminCreateUser(BaseModel):
    username: str
    email: str
    password: str
    full_name: str = ""
    student_id: str = ""
    # Bỏ qua role từ client — luôn ép là 'user'

@app.post("/api/admin/users")
async def admin_create_user(body: AdminCreateUser, admin: dict = Depends(_require_admin)):
    mysql = get_mysql()
    if mysql.get_user_by_username(body.username):
        raise HTTPException(409, f"Username '{body.username}' đã tồn tại")
    if mysql.get_user_by_email(body.email):
        raise HTTPException(409, f"Email '{body.email}' đã được dùng")

    uid = mysql.create_user(
        username=body.username,
        email=body.email,
        password_hash=hash_password(body.password),
        full_name=body.full_name,
        student_id=body.student_id,
        role="user",
    )
    if uid is None:
        raise HTTPException(500, "Lỗi tạo tài khoản")
    return {"message": "Tạo user thành công", "user_id": uid}

class AdminUpdateUser(BaseModel):
    username: Optional[str] = None
    email: Optional[str] = None
    password: Optional[str] = None
    full_name: Optional[str] = None
    student_id: Optional[str] = None
    face_registered: Optional[bool] = None
    # Không cho admin đổi role (nâng/hạ quyền) từ UI.

@app.put("/api/admin/users/{user_id}")
async def admin_update_user(user_id: int, body: AdminUpdateUser,
                            admin: dict = Depends(_require_admin)):
    mysql = get_mysql()
    user = mysql.get_user_by_id(user_id)
    if not user:
        raise HTTPException(404, "User không tồn tại")
    if (user.get("role") or "").lower() == "admin":
        raise HTTPException(403, "Không được chỉnh sửa tài khoản admin")

    updates = {}
    if body.username is not None:
        updates["username"] = body.username
    if body.email is not None:
        updates["email"] = body.email
    if body.password is not None:
        updates["password_hash"] = hash_password(body.password)
    if body.full_name is not None:
        updates["full_name"] = body.full_name
    if body.student_id is not None:
        updates["student_id"] = body.student_id
    if body.face_registered is not None:
        updates["face_registered"] = body.face_registered

    if not updates:
        return {"message": "Không có gì thay đổi"}

    mysql.update_user(user_id, **updates)
    return {"message": "Cập nhật thành công"}

@app.delete("/api/admin/users/{user_id}")
async def admin_delete_user(user_id: int, admin: dict = Depends(_require_admin)):
    mysql = get_mysql()
    user = mysql.get_user_by_id(user_id)
    if not user:
        raise HTTPException(404, "User không tồn tại")
    if (user.get("role") or "").lower() == "admin":
        raise HTTPException(403, "Không được xoá tài khoản admin")
    # Xoá embeddings trong Qdrant
    qdrant = get_qdrant()
    qdrant.delete_user_embeddings(user_id)
    # Xoá user trong MySQL
    mysql.delete_user(user_id)
    return {"message": f"Đã xoá user '{user['username']}'"}


# ══════════════════════════════════════════════════════════════════════════════
# API: ADMIN — Lịch thi (Exams) + Điểm danh
# ══════════════════════════════════════════════════════════════════════════════

class ExamCreate(BaseModel):
    subject: str
    exam_date: str           # "YYYY-MM-DD"
    start_time: str          # "HH:MM"
    end_time: str            # "HH:MM"
    room: str = ""
    note: str = ""
    student_ids: List[int] = []

class ExamUpdate(BaseModel):
    subject: Optional[str] = None
    exam_date: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    room: Optional[str] = None
    note: Optional[str] = None
    student_ids: Optional[List[int]] = None


# Giờ bắt đầu phải sớm hơn hiện tại ít nhất 30 phút (tức: start >= now + 30min)
EXAM_MIN_LEAD_MINUTES = 30

# Cửa sổ cho phép điểm danh: [start - X phút, end + X phút] của ngày thi
ATTENDANCE_WINDOW_MINUTES = 15


def _check_attendance_window(exam: Dict):
    """Chặn mark/unmark attendance nếu hiện tại nằm ngoài cửa sổ cho phép.
    Raise HTTPException(400) nếu sai thời gian."""
    import datetime as _dt
    try:
        d = _dt.date.fromisoformat(exam["exam_date"])
        st_str = exam["start_time"]
        en_str = exam["end_time"]
        st = _dt.time.fromisoformat(st_str if len(st_str) == 8 else st_str + ":00")
        en = _dt.time.fromisoformat(en_str if len(en_str) == 8 else en_str + ":00")
    except Exception:
        return
    now = _dt.datetime.now()
    window_start = _dt.datetime.combine(d, st) - _dt.timedelta(minutes=ATTENDANCE_WINDOW_MINUTES)
    window_end   = _dt.datetime.combine(d, en) + _dt.timedelta(minutes=ATTENDANCE_WINDOW_MINUTES)
    if now < window_start:
        raise HTTPException(
            400,
            f"Chưa tới giờ điểm danh. Chỉ được điểm danh từ "
            f"{window_start.strftime('%H:%M %d/%m/%Y')}",
        )
    if now > window_end:
        raise HTTPException(
            400,
            f"Đã quá giờ điểm danh (đóng lúc {window_end.strftime('%H:%M %d/%m/%Y')})",
        )


def _validate_exam_payload(subject: str, exam_date: str, start_time: str, end_time: str,
                           room: Optional[str] = None,
                           *, check_future: bool = False,
                           require_room: bool = False,
                           check_room_conflict: bool = False,
                           exclude_exam_id: Optional[int] = None):
    import datetime as _dt
    if not (subject or "").strip():
        raise HTTPException(400, "Thiếu môn thi")
    if require_room and not (room or "").strip():
        raise HTTPException(400, "Phòng thi là bắt buộc")
    try:
        d = _dt.date.fromisoformat(exam_date)
    except Exception:
        raise HTTPException(400, "Ngày thi không hợp lệ (YYYY-MM-DD)")
    try:
        st = _dt.time.fromisoformat(start_time if len(start_time) == 8 else start_time + ":00")
        en = _dt.time.fromisoformat(end_time   if len(end_time)   == 8 else end_time   + ":00")
    except Exception:
        raise HTTPException(400, "Giờ không hợp lệ (HH:MM)")
    if en <= st:
        raise HTTPException(400, "Giờ kết thúc phải sau giờ bắt đầu")
    if check_future:
        now = _dt.datetime.now()
        today = now.date()
        if d < today:
            raise HTTPException(400, "Ngày thi không được ở trong quá khứ")
        if d == today:
            start_dt = _dt.datetime.combine(d, st)
            min_start = now + _dt.timedelta(minutes=EXAM_MIN_LEAD_MINUTES)
            if start_dt < min_start:
                raise HTTPException(
                    400,
                    f"Với lịch thi trong hôm nay, giờ bắt đầu phải sớm hơn hiện tại "
                    f"ít nhất {EXAM_MIN_LEAD_MINUTES} phút "
                    f"(sớm nhất: {min_start.strftime('%H:%M')})",
                )
    if check_room_conflict and (room or "").strip():
        conflict = get_exams().find_room_conflict(
            room=room.strip(),
            exam_date=exam_date,
            start_time=start_time if len(start_time) == 8 else start_time + ":00",
            end_time=end_time if len(end_time) == 8 else end_time + ":00",
            exclude_id=exclude_exam_id,
        )
        if conflict:
            raise HTTPException(
                400,
                f"Phòng {room.strip()} đã có lịch '{conflict['subject']}' "
                f"từ {conflict['start_time'][:5]} đến {conflict['end_time'][:5]} "
                f"trong ngày này",
            )


def _raise_if_student_conflict(*, student_ids: List[int], exam_date: str,
                               start_time: str, end_time: str,
                               exclude_exam_id: Optional[int] = None):
    """Chặn gán 1 sinh viên vào 2 lịch thi trùng giờ (cùng ngày, chồng chéo)."""
    if not student_ids:
        return
    st = start_time if len(start_time) == 8 else start_time + ":00"
    en = end_time   if len(end_time)   == 8 else end_time   + ":00"
    conflicts = get_exams().find_student_conflicts(
        student_ids=student_ids,
        exam_date=exam_date,
        start_time=st, end_time=en,
        exclude_exam_id=exclude_exam_id,
    )
    if not conflicts:
        return
    # Gom theo sinh viên để thông báo gọn
    by_user: Dict[int, List[Dict]] = {}
    names: Dict[int, str] = {}
    for c in conflicts:
        by_user.setdefault(c["user_id"], []).append(c)
        names[c["user_id"]] = f"{c['student_id'] or c['username']} - {c['full_name'] or c['username']}"
    msgs = []
    for uid, items in by_user.items():
        clash = items[0]
        msgs.append(
            f"{names[uid]}: đã có '{clash['subject']}' "
            f"{clash['start_time'][:5]}–{clash['end_time'][:5]}"
        )
    raise HTTPException(400, "Xung đột lịch thi cho sinh viên: " + "; ".join(msgs))


def _filter_non_admin_ids(mysql: MySQLUserStore, ids: List[int]) -> List[int]:
    """Lọc bỏ id không tồn tại hoặc là admin."""
    out = []
    for uid in set(int(x) for x in ids):
        u = mysql.get_user_by_id(uid)
        if u and (u.get("role") or "").lower() != "admin":
            out.append(uid)
    return out


@app.get("/api/admin/exams")
async def admin_list_exams(admin: dict = Depends(_require_admin)):
    return {"exams": get_exams().list_exams()}


@app.get("/api/admin/exams/{exam_id}")
async def admin_get_exam(exam_id: int, admin: dict = Depends(_require_admin)):
    exam = get_exams().get_exam(exam_id)
    if not exam:
        raise HTTPException(404, "Lịch thi không tồn tại")
    return exam


@app.post("/api/admin/exams")
async def admin_create_exam(body: ExamCreate, admin: dict = Depends(_require_admin)):
    _validate_exam_payload(
        body.subject, body.exam_date, body.start_time, body.end_time,
        body.room,
        check_future=True,
        require_room=True,
        check_room_conflict=True,
    )
    mysql = get_mysql()
    student_ids = _filter_non_admin_ids(mysql, body.student_ids or [])
    _raise_if_student_conflict(
        student_ids=student_ids,
        exam_date=body.exam_date,
        start_time=body.start_time,
        end_time=body.end_time,
    )
    exam_id = get_exams().create_exam(
        subject=body.subject.strip(),
        exam_date=body.exam_date,
        start_time=body.start_time,
        end_time=body.end_time,
        room=(body.room or "").strip(),
        note=(body.note or "").strip(),
        student_ids=student_ids,
    )
    if exam_id is None:
        raise HTTPException(500, "Không tạo được lịch thi")
    return {"message": "Tạo lịch thi thành công", "exam_id": exam_id}


@app.put("/api/admin/exams/{exam_id}")
async def admin_update_exam(exam_id: int, body: ExamUpdate,
                            admin: dict = Depends(_require_admin)):
    exams = get_exams()
    existing = exams.get_exam(exam_id)
    if not existing:
        raise HTTPException(404, "Lịch thi không tồn tại")

    # Gộp lại thông tin lịch sau update để validate đầy đủ (bao gồm cả room)
    merged_subject   = body.subject    if body.subject    is not None else existing["subject"]
    merged_date      = body.exam_date  if body.exam_date  is not None else existing["exam_date"]
    merged_start     = body.start_time if body.start_time is not None else existing["start_time"]
    merged_end       = body.end_time   if body.end_time   is not None else existing["end_time"]
    merged_room      = body.room       if body.room       is not None else existing.get("room", "")

    if any(v is not None for v in (body.subject, body.exam_date, body.start_time,
                                   body.end_time, body.room)):
        _validate_exam_payload(
            merged_subject, merged_date, merged_start, merged_end, merged_room,
            require_room=True,
            check_room_conflict=True,
            exclude_exam_id=exam_id,
        )
    student_ids = None
    if body.student_ids is not None:
        student_ids = _filter_non_admin_ids(get_mysql(), body.student_ids)
        _raise_if_student_conflict(
            student_ids=student_ids,
            exam_date=merged_date,
            start_time=merged_start,
            end_time=merged_end,
            exclude_exam_id=exam_id,
        )

    exams.update_exam(
        exam_id,
        subject=body.subject.strip() if body.subject is not None else None,
        exam_date=body.exam_date,
        start_time=body.start_time,
        end_time=body.end_time,
        room=body.room.strip() if body.room is not None else None,
        note=body.note.strip() if body.note is not None else None,
        student_ids=student_ids,
    )
    return {"message": "Cập nhật lịch thi thành công"}


@app.delete("/api/admin/exams/{exam_id}")
async def admin_delete_exam(exam_id: int, admin: dict = Depends(_require_admin)):
    ok = get_exams().delete_exam(exam_id)
    if not ok:
        raise HTTPException(404, "Lịch thi không tồn tại")
    return {"message": "Đã xoá lịch thi"}


@app.post("/api/admin/exams/{exam_id}/attendance/{user_id}")
async def admin_mark_attendance(exam_id: int, user_id: int,
                                admin: dict = Depends(_require_admin)):
    """
    Đánh dấu 1 sinh viên có mặt trong lịch thi.
    Chỉ set attended_at lần đầu — các lần gọi lại giữ nguyên mốc cũ.
    Chỉ được thực hiện trong cửa sổ [start - 15m, end + 15m].
    """
    exam = get_exams().get_exam(exam_id)
    if not exam:
        raise HTTPException(404, "Lịch thi không tồn tại")
    _check_attendance_window(exam)
    result = get_exams().mark_attendance(exam_id, user_id)
    if result is None:
        raise HTTPException(404, "Sinh viên không có trong lịch thi này")
    return {
        "user_id": user_id,
        "attended_at": result["attended_at"],
        "first_time": result["first_time"],
    }


@app.delete("/api/admin/exams/{exam_id}/attendance/{user_id}")
async def admin_unmark_attendance(exam_id: int, user_id: int,
                                  admin: dict = Depends(_require_admin)):
    """Huỷ điểm danh 1 sinh viên (dành cho trường hợp nhận nhầm)."""
    exam = get_exams().get_exam(exam_id)
    if not exam:
        raise HTTPException(404, "Lịch thi không tồn tại")
    _check_attendance_window(exam)
    ok = get_exams().unmark_attendance(exam_id, user_id)
    if not ok:
        # Không có thay đổi — hoặc không thuộc lịch, hoặc chưa từng được đánh dấu
        raise HTTPException(404, "Không có bản ghi điểm danh để huỷ")
    return {"message": "Đã huỷ điểm danh", "user_id": user_id}


@app.get("/api/me/exams")
async def me_exams(current: dict = Depends(_get_current_user)):
    """Trả về danh sách lịch thi của sinh viên đang đăng nhập."""
    uid = int(current["sub"])
    return {"exams": get_exams().list_exams_for_user(uid)}


# ══════════════════════════════════════════════════════════════════════════════
# API: FACE REGISTRATION (giữ nguyên logic cũ)
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/api/face/start/{user_id}")
async def face_start(user_id: int, current: dict = Depends(_get_current_user)):
    """Bắt đầu face registration session cho user_id."""
    mysql = get_mysql()
    user = mysql.get_user_by_id(user_id)
    if not user:
        raise HTTPException(404, "User không tồn tại")

    # Chỉ admin hoặc chính user đó mới được đăng ký
    if current.get("role") != "admin" and int(current["sub"]) != user_id:
        raise HTTPException(403, "Không có quyền")

    # Không cho đăng ký khuôn mặt cho tài khoản admin
    if (user.get("role") or "").lower() == "admin":
        raise HTTPException(403, "Không đăng ký khuôn mặt cho tài khoản admin")

    # Không cho đăng ký lại khuôn mặt (user thường). Admin vẫn có thể mở session (hỗ trợ / chỉnh sửa).
    if user.get("face_registered") and current.get("role") != "admin":
        raise HTTPException(
            403,
            "Bạn đã đăng ký khuôn mặt. Không thể đăng ký lại.",
        )

    import uuid
    session_id = str(uuid.uuid4())

    embedder = get_embedder()
    reg = FaceRegistrationSession(
        user_id=str(user_id),
        hold_seconds=1.5,
        frames_per_step=5,
        face_app=embedder.app,
    )
    with _lock:
        _sessions[session_id] = reg
        _session_user[session_id] = user_id

    return {
        "session_id": session_id,
        "message": f"Bắt đầu đăng ký khuôn mặt cho {user['username']}",
        "steps": ["FRONT", "LEFT", "RIGHT", "UP", "DOWN"],
        "progress": reg.get_progress(),
    }


@app.post("/api/face/{session_id}/frame")
async def face_frame(session_id: str, frame: UploadFile = File(...)):
    with _lock:
        sess = _sessions.get(session_id)
        uid = _session_user.get(session_id)
    if not sess:
        raise HTTPException(404, "Session không tồn tại")
    if sess.is_done():
        return JSONResponse({"event": "done", "message": "Đã xong, gọi /finish"})

    raw = await frame.read()
    try:
        bgr = _decode_frame(raw)
    except ValueError as e:
        raise HTTPException(400, str(e))

    result = sess.process_frame(bgr)

    # ── Helper: check duplicate cho 1 direction ──
    def _check_direction_duplicate(finished_dir: str):
        reg_result = sess.get_result()
        cap = reg_result.captures.get(finished_dir)
        if not cap or not cap.frames:
            return None

        embedder = get_embedder()
        qdrant = get_qdrant()
        emb = embedder.extract_best(cap.frames)
        if emb is None:
            return None

        logger.info("[DUP-CHECK] Direction=%s, emb norm=%.4f", finished_dir, float(np.linalg.norm(emb)))

        # Same-person check vs FRONT
        if finished_dir != "FRONT":
            front_cap = reg_result.captures.get("FRONT")
            if front_cap and front_cap.frames:
                emb_front = embedder.extract_best(front_cap.frames)
                if emb_front is not None:
                    sim = _cosine_similarity(emb_front, emb)
                    logger.info("[DUP-CHECK] FRONT vs %s cosine=%.4f", finished_dir, sim)
                    if sim < SAME_PERSON_COSINE_THRESHOLD:
                        sess.redo_direction(finished_dir)
                        return JSONResponse({
                            "event": "mismatch_face",
                            "message": f"Khuôn mặt không khớp với ảnh chính diện (cosine={sim:.3f} < {SAME_PERSON_COSINE_THRESHOLD}). Vui lòng đăng ký lại góc {finished_dir}.",
                            "progress": sess.get_progress(),
                        })

        # DB duplicate check (exclude current user)
        raw_hits = qdrant.query_face_points(embedding=emb, top_k=10)
        if raw_hits:
            top_scores = [(getattr(h, "score", 0), h.payload.get("user_id", "?")) for h in raw_hits[:5]]
            logger.info("[DUP-CHECK] Direction=%s Qdrant top scores: %s", finished_dir, top_scores)

        dup = _embedding_duplicate_check(qdrant, emb, exclude_user_id=uid)
        if dup is not None:
            logger.warning("[DUP-CHECK] DUPLICATE! dir=%s match=%s score=%.4f",
                           finished_dir, dup['username'], dup['score'])
            sess.redo_direction(finished_dir)
            return JSONResponse({
                "event": "duplicate",
                "message": f"Khuôn mặt ở góc {finished_dir} đã tồn tại trong DB (match '{dup['username']}', score={dup['score']:.4f}). Vui lòng đăng ký lại góc {finished_dir}.",
                "progress": sess.get_progress(),
            })
        return None

    # Check trùng cho step_done
    if result.get("event") == "step_done":
        finished_dir = result.get("finished_direction")
        if finished_dir:
            dup_resp = _check_direction_duplicate(finished_dir)
            if dup_resp is not None:
                return dup_resp

    # Check trùng cho góc cuối (done)
    if result.get("event") == "done":
        dup_resp = _check_direction_duplicate("DOWN")
        if dup_resp is not None:
            return dup_resp

    return JSONResponse(result)


@app.post("/api/face/{session_id}/finish")
async def face_finish(session_id: str):
    """Lưu embedding vào Qdrant + set face_registered trong MySQL."""
    with _lock:
        if session_id in _face_finish_cache:
            return _face_finish_cache[session_id]
        sess = _sessions.get(session_id)
        uid = _session_user.get(session_id)
    if not sess or uid is None:
        raise HTTPException(404, "Session không tồn tại")
    if not sess.is_done():
        raise HTTPException(400, "Chưa hoàn thành đăng ký khuôn mặt")

    reg_result = sess.get_result()
    embedder = get_embedder()
    qdrant = get_qdrant()
    mysql = get_mysql()

    # Safety: same-person check
    front_cap = reg_result.captures.get("FRONT")
    if not front_cap or not front_cap.frames:
        raise HTTPException(400, "Thiếu góc FRONT")
    emb_front = embedder.extract_best(front_cap.frames)
    if emb_front is None:
        raise HTTPException(500, "Không extract được embedding FRONT")
    for d, cap in reg_result.captures.items():
        if d == "FRONT" or not cap or not cap.frames:
            continue
        emb_d = embedder.extract_best(cap.frames)
        if emb_d is None:
            continue
        sim = _cosine_similarity(emb_front, emb_d)
        if sim < SAME_PERSON_COSINE_THRESHOLD:
            raise HTTPException(409, f"Góc {d} không khớp FRONT (cosine={sim:.3f})")

    # Final duplicate check
    user_emb = build_user_embedding(reg_result.captures, embedder)
    if user_emb is None:
        raise HTTPException(500, "Không extract được embedding")
    dup = _embedding_duplicate_check(qdrant, user_emb, exclude_user_id=uid)
    if dup is not None:
        raise HTTPException(409, f"Khuôn mặt đã tồn tại (match '{dup['username']}', score={dup['score']:.3f})")

    # Xóa embeddings cũ nếu có
    qdrant.delete_user_embeddings(uid)

    # Lưu embeddings mới
    n_points = qdrant.save_user_embeddings(
        user_id=uid,
        captures=reg_result.captures,
        embedder=embedder,
    )
    if n_points == 0:
        raise HTTPException(500, "Không extract được embedding")

    # Set face_registered = True
    mysql.set_face_registered(uid, True)
    logger.info("Saved %d points for user_id=%s, face_registered=True", n_points, uid)

    user = mysql.get_user_by_id(uid)
    out = {
        "message": f"Đăng ký khuôn mặt thành công cho {user['username']}!",
        "user": {
            "id": uid,
            "username": user["username"],
            "face_registered": True,
        },
    }

    # Cleanup + cache (idempotent /finish)
    with _lock:
        _face_finish_cache[session_id] = out
        _sessions.pop(session_id, None)
        _session_user.pop(session_id, None)
    sess.close()

    return out


# ══════════════════════════════════════════════════════════════════════════════
# API: FACE VERIFY
# ══════════════════════════════════════════════════════════════════════════════

def _face_payload(
    bbox: list,
    *,
    match: bool,
    score: Optional[float] = None,
    user: Optional[dict] = None,
    message: str = "",
) -> dict:
    out = {
        "bbox": bbox,
        "match": match,
        "score": score,
        "user": user,
        "message": message,
    }
    return out


@app.get("/api/face/verify/device")
def face_verify_device():
    """Trả về thiết bị đang chạy của InsightFace + FAS để hiển thị overlay."""
    embedder = get_embedder()
    fas = get_fas()
    return {
        "insightface": _infer_insightface_device(embedder.app),
        "fas": _infer_fas_device(fas),
    }


@app.post("/api/face/verify")
async def face_verify(frame: UploadFile = File(...)):
    """
    Nhiều người (tối đa VERIFY_MAX_FACES_PER_FRAME), det_score cao trước.
    Mỗi mặt: FAS → embedding → Qdrant vote + cosine hiển thị.
    Trả `faces`: [{ bbox, match, score, user?, message? }, ...] và match tổng (có ít nhất 1 khớp).
    """
    raw = await frame.read()
    try:
        bgr = _decode_frame(raw)
    except ValueError as e:
        raise HTTPException(400, str(e))

    from src.utility import get_crop_face

    embedder = get_embedder()
    fas = get_fas()
    qdrant = get_qdrant()
    mysql = get_mysql()

    faces = embedder.app.get(bgr)
    if not faces:
        return {"match": False, "message": "Không phát hiện khuôn mặt", "faces": []}

    ordered = sorted(
        faces,
        key=lambda f: float(getattr(f, "det_score", 0.0)),
        reverse=True,
    )[:VERIFY_MAX_FACES_PER_FRAME]

    out_faces: List[dict] = []
    any_match = False
    now_ts = time.time()

    def _cache_lookup(emb_norm_vec):
        """Trả payload cache nếu có embedding tương đồng >= VERIFY_CACHE_SIM và còn TTL."""
        best_sim = -1.0
        best_payload = None
        with _verify_cache_lock:
            fresh = []
            for ts, cemb, payload in _verify_cache:
                if now_ts - ts > VERIFY_CACHE_TTL:
                    continue
                fresh.append((ts, cemb, payload))
                sim = float(np.dot(emb_norm_vec, cemb))
                if sim > best_sim:
                    best_sim = sim
                    best_payload = payload
            _verify_cache[:] = fresh
        if best_payload is not None and best_sim >= VERIFY_CACHE_SIM:
            return best_payload
        return None

    def _cache_store(emb_norm_vec, payload):
        with _verify_cache_lock:
            if len(_verify_cache) >= VERIFY_CACHE_MAX:
                _verify_cache.pop(0)
            _verify_cache.append((now_ts, emb_norm_vec.copy(), payload))

    for face in ordered:
        bb = face.bbox
        bbox = [float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])]

        emb = face.embedding
        n = float(np.linalg.norm(emb))
        emb_norm = (emb / n if n > 1e-9 else emb).astype(np.float32)

        cached = _cache_lookup(emb_norm)
        if cached is not None:
            payload = dict(cached)
            payload["bbox"] = bbox
            if payload.get("match"):
                any_match = True
            out_faces.append(payload)
            continue

        # Occlusion check (mask / kính đen) — từ chối trước khi FAS/Qdrant
        try:
            occ, reasons, _m = detect_occlusion(bgr, face)
        except Exception:
            occ, reasons = False, []
        if occ:
            payload = _face_payload(
                bbox,
                match=False,
                score=None,
                user=None,
                message=describe_reasons(reasons),
            )
            payload["occluded"] = True
            payload["occlusion_reasons"] = reasons
            out_faces.append(payload)
            continue

        i27 = get_crop_face(bgr, face.bbox, 2.7)
        i40 = get_crop_face(bgr, face.bbox, 4.0)
        label, fas_s = fas.predict(i27, i40)
        if not (label == VERIFY_FAS_LABEL_REAL and fas_s > VERIFY_FAS_SCORE_MIN):
            payload = _face_payload(
                bbox,
                match=False,
                score=None,
                user=None,
                message="Không phải khuôn mặt thật (ảnh 2D / in giấy) hoặc chất lượng không đủ",
            )
            out_faces.append(payload)
            _cache_store(emb_norm, payload)
            continue

        uid, q_score = qdrant.match_face_like_demo_detailed(
            emb_norm,
            top_k=VERIFY_QDRANT_TOP_K,
            score_threshold=VERIFY_QDRANT_SCORE_MIN,
        )
        if uid is None:
            payload = _face_payload(
                bbox,
                match=False,
                score=q_score,
                user=None,
                message="Không tìm thấy khuôn mặt trong hệ thống",
            )
            out_faces.append(payload)
            _cache_store(emb_norm, payload)
            continue

        user = mysql.get_user_by_id(int(uid))
        if not user:
            payload = _face_payload(
                bbox,
                match=False,
                score=q_score,
                user=None,
                message="User không tồn tại",
            )
            out_faces.append(payload)
            _cache_store(emb_norm, payload)
            continue

        any_match = True
        payload = _face_payload(
            bbox,
            match=True,
            score=q_score,
            user={
                "full_name": (user.get("full_name") or user.get("username") or "").strip(),
                "student_id": (user.get("student_id") or "").strip(),
            },
            message="",
        )
        out_faces.append(payload)
        _cache_store(emb_norm, payload)

    return {
        "match": any_match,
        "faces": out_faces,
    }
