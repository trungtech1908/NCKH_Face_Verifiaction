"""
FastAPI Routes
--------------
Auth:
  POST /api/register/info      — bước 1: nhập username/email/password
  POST /api/register/start     — bước 2: bắt đầu face registration session
  POST /api/register/{sid}/frame — bước 3: gửi frame
  POST /api/register/{sid}/finish — bước 4: extract embedding + lưu Qdrant

  POST /api/login/password     — đăng nhập bằng password
  POST /api/login/face         — đăng nhập bằng khuôn mặt

  GET  /api/me                 — thông tin user (cần JWT)
"""

import uuid, io, logging, threading, time
import numpy as np
import cv2

from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, Header
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from pydantic import BaseModel, EmailStr
from typing import Optional, Dict

from core.registration import FaceRegistrationSession
from core.embedding import FaceEmbedder
from storage.qdrant_store import QdrantFaceStore
from api.auth import (hash_password, verify_password,
                      create_access_token, decode_token, new_user_id)
import config

logger = logging.getLogger(__name__)

app = FastAPI(title="Face Auth API")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ── Singletons ─────────────────────────────────────────────────────────────
_store: Optional[QdrantFaceStore] = None
_embedder: Optional[FaceEmbedder] = None
_embedder_lock = threading.Lock()

def get_store() -> QdrantFaceStore:
    global _store
    if _store is None:
        _store = QdrantFaceStore()
    return _store

def get_embedder() -> FaceEmbedder:
    global _embedder
    if _embedder is None:
        with _embedder_lock:
            if _embedder is None:
                _embedder = FaceEmbedder()
    return _embedder

# ── Session store ───────────────────────────────────────────────────────────
_sessions: Dict[str, FaceRegistrationSession] = {}
_pending:  Dict[str, dict] = {}   # sid → user info (chờ face xong)
_lock = threading.Lock()

# ── Helpers ─────────────────────────────────────────────────────────────────
def _decode_frame(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Cannot decode image")
    return frame

def _get_current_user(authorization: Optional[str] = Header(None)) -> dict:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing token")
    token = authorization.split(" ", 1)[1]
    payload = decode_token(token)
    if not payload:
        raise HTTPException(401, "Invalid or expired token")
    return payload

# ── Pages ────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

# ── API: Register ─────────────────────────────────────────────────────────────
class UserInfoRequest(BaseModel):
    username: str
    email:    str
    password: str

@app.post("/api/register/info")
async def register_info(body: UserInfoRequest):
    """Bước 1: kiểm tra username/email không trùng, lưu tạm."""
    store = get_store()
    if store.get_user_by_username(body.username):
        raise HTTPException(409, f"Username '{body.username}' đã tồn tại")
    if store.get_user_by_email(body.email):
        raise HTTPException(409, f"Email '{body.email}' đã được dùng")

    session_id = str(uuid.uuid4())
    with _lock:
        _pending[session_id] = {
            "user_id":  new_user_id(),
            "username": body.username,
            "email":    body.email,
            "password": hash_password(body.password),
        }
    return {"session_id": session_id,
            "message": "Thông tin hợp lệ. Tiến hành đăng ký khuôn mặt."}


@app.post("/api/register/start/{session_id}")
async def register_start(session_id: str):
    """Bước 2: khởi tạo face registration session."""
    with _lock:
        info = _pending.get(session_id)
    if not info:
        raise HTTPException(404, "Session không tồn tại hoặc đã hết hạn")

    reg = FaceRegistrationSession(
        user_id=info["user_id"],
        hold_seconds=1.5,
        frames_per_step=5,
    )
    with _lock:
        _sessions[session_id] = reg

    return {"message": "Bắt đầu đăng ký khuôn mặt",
            "steps": ["FRONT","LEFT","RIGHT","UP","DOWN"],
            "progress": reg.get_progress()}


@app.post("/api/register/{session_id}/frame")
async def register_frame(session_id: str,
                         frame: UploadFile = File(...)):
    with _lock:
        sess = _sessions.get(session_id)
    if not sess:
        raise HTTPException(404, "Session không tồn tại")
    if sess.is_done():
        return JSONResponse({"event":"done","message":"Đã xong, gọi /finish"})

    raw = await frame.read()
    try:
        bgr = _decode_frame(raw)
    except ValueError as e:
        raise HTTPException(400, str(e))

    result = sess.process_frame(bgr)
    return JSONResponse(result)


@app.post("/api/register/{session_id}/finish")
async def register_finish(session_id: str):
    """Bước 4: extract embedding + lưu Qdrant."""
    with _lock:
        sess = _sessions.get(session_id)
        info = _pending.get(session_id)
    if not sess or not info:
        raise HTTPException(404, "Session không tồn tại")
    if not sess.is_done():
        raise HTTPException(400, "Chưa hoàn thành đăng ký khuôn mặt")

    reg_result = sess.get_result()
    embedder   = get_embedder()
    store      = get_store()

    user_payload = {
        "username":   info["username"],
        "email":      info["email"],
        "password":   info["password"],
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    n_points = store.save_user_embeddings(
        user_id=info["user_id"],
        user_payload=user_payload,
        captures=reg_result.captures,
        embedder=embedder,
    )
    if n_points == 0:
        raise HTTPException(500, "Không extract được embedding khuôn mặt")
    logger.info("Saved %d points for user=%s", n_points, info["user_id"])

    # Cleanup
    with _lock:
        _sessions.pop(session_id, None)
        _pending.pop(session_id, None)
    sess.close()

    token = create_access_token(info["user_id"], info["username"])
    return {
        "message": f"Đăng ký thành công! Chào mừng {info['username']}",
        "token":   token,
        "user":    {"user_id": info["user_id"], "username": info["username"],
                    "email": info["email"]},
    }


# ── API: Login ───────────────────────────────────────────────────────────────
class PasswordLoginRequest(BaseModel):
    username: str
    password: str

@app.post("/api/login/password")
async def login_password(body: PasswordLoginRequest):
    store = get_store()
    user  = store.get_user_by_username(body.username)
    if not user:
        raise HTTPException(401, "Username không tồn tại")
    if not verify_password(body.password, user["password"]):
        raise HTTPException(401, "Sai mật khẩu")

    token = create_access_token(user["user_id"], user["username"])
    return {
        "message": f"Đăng nhập thành công!",
        "token":   token,
        "user":    {"user_id": user["user_id"], "username": user["username"],
                    "email": user["email"]},
    }



# ── API: Me ───────────────────────────────────────────────────────────────────
@app.get("/api/me")
async def get_me(current: dict = Depends(_get_current_user)):
    store = get_store()
    user  = store.get_user_by_id(current["sub"])
    if not user:
        raise HTTPException(404, "User không tồn tại")
    return {
        "user_id":    user["user_id"],
        "username":   user["username"],
        "email":      user["email"],
        "created_at": user.get("created_at"),
    }
