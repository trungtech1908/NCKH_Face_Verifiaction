# Face Recognition System With Face Anti-Spoofing

## Giới Thiệu

Đây là hệ thống nhận diện khuôn mặt tích hợp Face Anti-Spoofing nhằm tăng cường tính bảo mật và hạn chế các hình thức giả mạo như:

- Ảnh in
- Video phát lại
- Hiển thị khuôn mặt qua màn hình điện thoại

Hệ thống sử dụng mô hình nhận diện khuôn mặt kết hợp với mô hình chống giả mạo để xác minh người dùng theo thời gian thực.

---

## Chức Năng Chính

- Phát hiện khuôn mặt
- Căn chỉnh khuôn mặt
- Nhận diện danh tính
- Face Anti-Spoofing
- Lưu trữ embedding
- Tìm kiếm khuôn mặt bằng vector database Qdrant

---

## Công Nghệ Sử Dụng

- Python
- MySQL
- Qdrant
- Deep Learning
- Face Recognition
- Face Anti-Spoofing

---

## Các Bước Chạy

### 1. Cài đặt thư viện

```bash
pip install -r requirement.txt
```

---

### 2. Thiết lập biến môi trường

Tạo file `.env` dựa trên file `.env.example`.

Cấu hình đầy đủ các thông tin cần thiết bao gồm:

- MySQL
- Qdrant

---

### 3. Chạy project

```bash
python main.py
```