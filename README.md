# Hệ thống Xác minh Khuôn mặt Sinh viên (NCKH)

Hệ thống điểm danh và xác minh sinh viên bằng nhận diện khuôn mặt thời gian thực.
Stack: **FastAPI + InsightFace (buffalo_l) + MiniFASNet (anti‑spoof) + Qdrant Cloud + MySQL**.

---

## 1. Yêu cầu hệ thống

| Thành phần  | Phiên bản đề xuất                                |
| ----------- | ------------------------------------------------ |
| OS          | Linux (đã test trên Ubuntu 22.04), Windows cũng chạy |
| Python      | 3.9 – 3.11                                       |
| MySQL       | 8.0+                                             |
| GPU (tùy chọn) | NVIDIA + CUDA 11.8/12.x (tăng tốc InsightFace & FAS) |
| Camera      | Webcam hỗ trợ WebRTC (Chrome/Edge/Firefox khuyến nghị) |
| Mạng        | Cần internet để gọi Qdrant Cloud                 |

> Không có GPU vẫn chạy được bằng CPU (FPS thấp hơn).

---

## 2. Chuẩn bị mã nguồn và Python env

```bash
# 1. Clone
git clone <repo-url> NCKH_Face_Verifiaction
cd NCKH_Face_Verifiaction

# 2. Tạo & kích hoạt virtualenv
python3 -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows PowerShell

# 3. Cài thư viện
pip install --upgrade pip
pip install -r requirements.txt
```

### 2.1. (Tùy chọn) Kích hoạt GPU cho InsightFace / FAS

Mặc định `onnxruntime` trong `requirements.txt` là bản CPU. Nếu có GPU NVIDIA:

```bash
pip uninstall onnxruntime -y
pip install onnxruntime-gpu
```

Kiểm tra nhanh trong Python:

```python
import onnxruntime as ort
print(ort.get_available_providers())
# Mong đợi: ['CUDAExecutionProvider', 'CPUExecutionProvider', ...]
```

Nếu log server hiện `providers=['CUDAExecutionProvider', ...]` là đã dùng GPU.

---

## 3. Cài đặt & cấu hình MySQL

Hệ thống dùng MySQL để lưu tài khoản người dùng (bảng `users`).
Server sẽ **tự tạo database và bảng** ở lần chạy đầu tiên, bạn chỉ cần có sẵn MySQL
và một tài khoản có quyền `CREATE DATABASE`.

### 3.1. Cài MySQL Server

**Ubuntu / Debian**

```bash
sudo apt update
sudo apt install -y mysql-server
sudo systemctl enable --now mysql
```

**Windows**: tải MySQL Installer tại <https://dev.mysql.com/downloads/installer/> → chọn bản Community → cài với mật khẩu root mong muốn.

### 3.2. Kiểm tra dịch vụ

```bash
sudo systemctl status mysql         # Linux
# hoặc
mysql --version
```

### 3.3. (Khuyến nghị) Tạo user riêng cho app

Đăng nhập root:

```bash
sudo mysql                # Ubuntu thường dùng auth socket
# hoặc
mysql -u root -p
```

Trong MySQL shell:

```sql
CREATE USER 'face_app'@'localhost' IDENTIFIED BY 'your-strong-password';
GRANT ALL PRIVILEGES ON face_verification.* TO 'face_app'@'localhost';
-- Cần quyền tạo database khi lần đầu chạy:
GRANT CREATE ON *.* TO 'face_app'@'localhost';
FLUSH PRIVILEGES;
EXIT;
```

> Nếu lười, dùng `root` cũng chạy được — chỉ cần đặt `MYSQL_USER=root` và `MYSQL_PASSWORD` đúng trong `.env`.

### 3.4. Kiểm tra kết nối

```bash
mysql -h 127.0.0.1 -P 3306 -u face_app -p
```

Gõ được vào prompt `mysql>` là xong. Thoát bằng `EXIT;`.

> **Lưu ý**: không cần tự tạo database `face_verification` hay bảng `users`. Khi chạy `python main.py` lần đầu, file `storage/mysql_store.py → init_database()` sẽ tự tạo và seed tài khoản admin mặc định.

---

## 4. Chuẩn bị Qdrant Cloud

1. Tạo tài khoản miễn phí tại <https://cloud.qdrant.io>.
2. Tạo 1 cluster → copy **Endpoint URL** và **API Key**.
3. Collection (ví dụ `NCKH_Face_Verification`) sẽ được tự tạo khi chạy lần đầu (vector 512‑d, distance COSINE).

---

## 5. Tạo file `.env`

Copy mẫu và điền thông tin thực tế:

```bash
cp env.example .env
```

Nội dung `.env` cơ bản:

```env
# Qdrant
URL_QDRANT=https://xxxxxxxx.cloud.qdrant.io:6333
API_QDRANT=your-qdrant-api-key
QDRANT_COLLECTION=NCKH_Face_Verification

# JWT
JWT_SECRET=change-this-secret-in-production
JWT_EXPIRE_HOURS=24

# MySQL
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=face_app
MYSQL_PASSWORD=your-strong-password
MYSQL_DATABASE=face_verification

# Thresholds (giữ mặc định nếu không rõ)
FACE_SIMILARITY_THRESHOLD=0.45
ANTI_SPOOF_SCORE_THRESHOLD=0.85
ANTI_SPOOF_CONSISTENT_FRAMES=3
ANTI_SPOOF_STATIC_THRESHOLD=2.0
ANTI_SPOOF_STATIC_FRAMES=5
```

> `URL_QDRANT` và `API_QDRANT` **bắt buộc**, thiếu sẽ báo lỗi `RuntimeError` khi khởi động.

---

## 6. Chạy server

```bash
source .venv/bin/activate          # nếu chưa kích hoạt venv
python main.py
```

Lần chạy đầu, log sẽ hiện:

```
InsightFace buffalo_l loaded (providers=[...])
MySQL database initialized
Created default admin account (admin / admin123)
Uvicorn running on http://0.0.0.0:8000
```

Mở trình duyệt: **<http://localhost:8000>**

---

## 7. Tài khoản mặc định

| Username | Password  | Role  | Ghi chú                                |
| -------- | --------- | ----- | -------------------------------------- |
| `admin`  | `admin123`| admin | Được tự tạo lần đầu chạy. **Đổi mật khẩu ngay trong production.** |

Để đổi mật khẩu admin nhanh, vào MySQL:

```sql
USE face_verification;
-- sau đó đăng nhập /login với tk admin → thay qua UI, hoặc update password_hash bằng bcrypt hash.
```

---

## 8. Luồng sử dụng

### 8.1. Người dùng (User)

1. `/register` → tạo tài khoản sinh viên.
2. `/login` → đăng nhập, hệ thống chuyển tới `/user`.
3. Trong dashboard → **Đăng ký khuôn mặt** → quay 5 góc: `FRONT → LEFT → RIGHT → UP → DOWN`. Mỗi góc thành công sẽ có toast + dấu tích ✅.
4. Sau khi đủ 5 góc, embedding được đẩy lên Qdrant và tài khoản được đánh dấu `face_registered = true`.

### 8.2. Admin

1. Đăng nhập `/login` bằng `admin/admin123` → chuyển tới `/admin`.
2. Tab **Quản lý Sinh viên**: thêm / sửa / xoá **tài khoản user thường**. Không thao tác được tài khoản admin khác (chặn từ backend).
3. Nhấn 📷 trên dòng sinh viên để đăng ký khuôn mặt thay cho họ.
4. Tab **Xác minh Khuôn mặt**: bật camera → khung hình vẽ bbox + họ tên + MSSV + cosine score; chạy liên tục theo FPS tối đa của máy (hiển thị cả FPS và thiết bị GPU/CPU).

---

## 9. Cấu trúc thư mục

```
.
├── main.py                    # entry point uvicorn
├── config.py                  # đọc .env
├── api/
│   ├── auth.py                # JWT + bcrypt
│   └── routes.py              # Toàn bộ endpoint FastAPI
├── core/
│   ├── embedding.py           # InsightFace wrapper
│   ├── anti_spoof.py          # FAS pipeline
│   ├── anti_spoof_predict.py  # MiniFASNet inference
│   ├── registration.py        # Logic 5 góc
│   ├── registration_FAS.py    # Registration + anti-spoof
│   └── head_pose.py
├── storage/
│   ├── mysql_store.py         # MySQL users table
│   └── qdrant_store.py        # Qdrant face vectors
├── external/
│   └── anti_spoofing/         # Weights MiniFASNet (.pth)
├── templates/                 # Jinja2 HTML
│   ├── index.html
│   ├── login.html, register.html
│   ├── admin.html
│   ├── user_dashboard.html
│   └── face_register.html
├── static/                    # css / js / ảnh
├── requirements.txt
├── env.example
└── README.md
```

---

## 10. API chính

| Method | Endpoint                              | Yêu cầu | Mô tả                                     |
| ------ | ------------------------------------- | ------- | ----------------------------------------- |
| POST   | `/api/register`                       | -       | Đăng ký tài khoản user                    |
| POST   | `/api/login`                          | -       | Đăng nhập, trả JWT                        |
| GET    | `/api/me`                             | JWT     | Thông tin user hiện tại                   |
| GET    | `/api/admin/users`                    | Admin   | Danh sách **user** (lọc bỏ admin)         |
| POST   | `/api/admin/users`                    | Admin   | Tạo user (role luôn = `user`)             |
| PUT    | `/api/admin/users/{id}`               | Admin   | Sửa user thường (chặn sửa admin)          |
| DELETE | `/api/admin/users/{id}`               | Admin   | Xoá user + embeddings (chặn xoá admin)    |
| POST   | `/api/face/start/{user_id}`           | JWT     | Mở phiên đăng ký khuôn mặt                |
| POST   | `/api/face/{session_id}/frame`        | -       | Gửi frame camera                          |
| POST   | `/api/face/{session_id}/finish`       | -       | Đẩy embedding 5 góc lên Qdrant            |
| POST   | `/api/face/verify`                    | JWT     | Xác minh 1 frame (nhiều khuôn mặt)        |
| GET    | `/api/face/verify/device`             | JWT     | Thiết bị đang chạy (CPU / CUDA)           |

---

## 11. Troubleshooting

**`RuntimeError: Thiếu URL_QDRANT hoặc API_QDRANT trong file .env`**
→ Kiểm tra lại `.env` đã đặt đúng tên biến, và đã `cp env.example .env`.

**`mysql.connector.errors.DatabaseError: Access denied for user ...`**
→ Sai `MYSQL_USER` / `MYSQL_PASSWORD`, hoặc user chưa có quyền. Dùng tài khoản root hoặc cấp quyền theo mục 3.3.

**`Can't connect to MySQL server on 'localhost'`**
→ MySQL chưa chạy: `sudo systemctl start mysql`. Kiểm tra `MYSQL_HOST` và `MYSQL_PORT`.

**Camera đen hoặc không bật**
→ Trình duyệt chặn quyền camera. Truy cập qua `http://localhost:8000` (không phải IP LAN) hoặc bật HTTPS. Chrome: `Settings → Privacy → Site settings → Camera`.

**FPS thấp ~1‑3**
→ Đang chạy CPU. Cài `onnxruntime-gpu` (mục 2.1) hoặc giảm độ phân giải camera.

**Verify luôn trả "Không khớp"**
→ User chưa đăng ký khuôn mặt xong (chưa đủ 5 góc), hoặc `FACE_SIMILARITY_THRESHOLD` quá cao. Thử hạ xuống `0.40`.

**Reset dữ liệu**
→ Xoá collection Qdrant bằng `python delete_qdrant.py` và `DROP DATABASE face_verification;` trong MySQL, sau đó chạy lại server.

---

## 12. Kiến trúc dữ liệu

```
MySQL (users)                    Qdrant Cloud (face_verification)
┌─────────────────────┐          ┌─────────────────────────┐
│ id (PK)             │          │ vector: float32[512]    │
│ username UNIQUE     │          │ payload:                │
│ email UNIQUE        │◄────────►│   { user_id: <int> }    │
│ password_hash       │ user_id  │ distance: COSINE        │
│ full_name, studentId│          └─────────────────────────┘
│ role (admin|user)   │
│ face_registered     │
│ created_at, updated │
└─────────────────────┘
```

- MySQL giữ thông tin định danh tài khoản.
- Qdrant chỉ giữ vector + `user_id` để khi tìm được hit thì tra về MySQL lấy họ tên / MSSV.

---

## 13. Giấy phép & trích dẫn

Dự án phục vụ nghiên cứu khoa học (NCKH). Vui lòng trích dẫn nguồn nếu sử dụng lại mã nguồn / ý tưởng.
