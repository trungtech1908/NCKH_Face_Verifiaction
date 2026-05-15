import os
import cv2
from insightface.app import FaceAnalysis

# Khởi tạo
app = FaceAnalysis(name="buffalo_l", allowed_modules=['detection'])
app.prepare(ctx_id=0, det_size=(640, 640))

# Đường dẫn bạn đang dùng trong main script
dataset_path = "./test/Casia-fasd/test_img/test_img/color"

print(f"--- Đang kiểm tra thư mục: {dataset_path} ---")

# Lấy danh sách 5 file đầu tiên
files_found = []
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            files_found.append(os.path.join(root, file))
            if len(files_found) >= 5: break
    if len(files_found) >= 5: break

if not files_found:
    print("❌ Lỗi: Không tìm thấy bất kỳ file ảnh nào (.jpg, .png) trong đường dẫn này!")
else:
    for p in files_found:
        print(f"\n🔍 Thử đọc file: {p}")
        img = cv2.imread(p)
        if img is None:
            print("  -> ❌ OpenCV không thể đọc ảnh này (imread trả về None)")
        else:
            faces = app.get(img)
            print(f"  -> ✅ Đã đọc được ảnh ({img.shape})")
            print(f"  -> 👤 Số mặt tìm thấy: {len(faces)}")