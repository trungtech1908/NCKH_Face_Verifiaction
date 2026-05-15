import os
import cv2
import time
import torch
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from insightface.app import FaceAnalysis
from src.utility import get_crop_face
from src.model_lib.MiniFASNet import MiniFASNetV2, MiniFASNetV1SE

# ================== CONFIG ==================
MODEL_DIR = "./external/anti_spoofing"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESHOLD = 0.85
WEIGHT_27 = 0.6
WEIGHT_40 = 0.4


# ================== MODEL LOADING ==================
def load_model(model_name, model_class):
    model_path = os.path.join(MODEL_DIR, model_name)
    if not os.path.exists(model_path):
        print(f"Lỗi: Không tìm thấy file trọng số tại {model_path}")
        return None

    model = model_class(conv6_kernel=5, num_classes=3)
    state_dict = torch.load(model_path, map_location=DEVICE)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.to(DEVICE)
    model.eval()
    return model


print(f"--- Đang khởi tạo mô hình trên thiết bị: {DEVICE} ---")
model_v2 = load_model("2.7_80x80_MiniFASNetV2.pth", MiniFASNetV2)
model_v1se = load_model("4_0_0_80x80_MiniFASNetV1SE.pth", MiniFASNetV1SE)

# ================== INSIGHTFACE SETUP ==================
# Sử dụng det_size lớn hơn (1280) để nhận diện mặt tốt hơn trong các dataset phức tạp
app = FaceAnalysis(name="buffalo_l", allowed_modules=['detection'])
app.prepare(ctx_id=0, det_thresh=0.4, det_size=(320, 320))


# ================== UTILITY FUNCTIONS ==================
def preprocess(img):
    if img is None or img.size == 0:
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_res = cv2.resize(img_rgb, (80, 80))
    img_data = img_res.astype('float32')
    img_data = img_data.transpose((2, 0, 1))
    return torch.tensor(img_data).to(DEVICE)


def predict_liveness(i27, i40):
    t27 = preprocess(i27)
    t40 = preprocess(i40)
    if t27 is None or t40 is None:
        return 0.0

    with torch.no_grad():
        out27 = model_v2(t27.unsqueeze(0))
        out40 = model_v1se(t40.unsqueeze(0))
        score_27 = torch.softmax(out27, dim=1)[0, 1].item()
        score_40 = torch.softmax(out40, dim=1)[0, 1].item()
        # Công thức Fusion chuẩn của đồ án: 0.6*Scale2.7 + 0.4*Scale4.0
        final_score = score_27 * WEIGHT_27 + score_40 * WEIGHT_40
        return final_score


def get_ground_truth(path, dataset_name):
    path_lower = path.lower()
    if "celeba" in dataset_name.lower():
        # CelebA-Spoof: folder 'live' là 1, 'spoof' là 0
        return 1 if "/live/" in path_lower else 0
    else:
        # CASIA-FASD: Folder 1, 2, 3 là Real; 7, 8, 9... là Attack
        # Kiểm tra xem đường dẫn có chứa các thư mục con đánh số của ảnh thật không
        real_folders = ["/1/", "/2/", "/3/", "real"]
        if any(folder in path_lower for folder in real_folders):
            return 1
        return 0


def evaluate_dataset(dataset_path, dataset_name):
    image_paths = []
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, file))

    processed_images = 0
    predictions = []
    ground_truths = []

    start_time = time.time()

    # Sử dụng tqdm để hiển thị thanh tiến trình
    for img_path in tqdm(image_paths, desc=f"Đang quét {dataset_name}", unit="img"):
        img = cv2.imread(img_path)
        if img is None: continue

        faces = app.get(img)
        if not faces: continue

        face = max(faces, key=lambda f: getattr(f, 'det_score', 0.0))
        bbox = face.bbox

        i27 = get_crop_face(img, bbox, 2.7)
        i40 = get_crop_face(img, bbox, 4.0)

        final_score = predict_liveness(i27, i40)
        pred = 1 if final_score >= THRESHOLD else 0
        gt = get_ground_truth(img_path, dataset_name)

        predictions.append(pred)
        ground_truths.append(gt)
        processed_images += 1

    total_time = time.time() - start_time
    fps = processed_images / total_time if total_time > 0 else 0

    if not predictions:
        return {"total": len(image_paths), "proc": 0, "APCER": 0, "BPCER": 0, "ACER": 0, "FPS": 0}

    # Tính toán chỉ số chuyên ngành: APCER, BPCER, ACER
    total_fake = ground_truths.count(0)
    total_real = ground_truths.count(1)

    fake_as_real = sum(1 for p, g in zip(predictions, ground_truths) if p == 1 and g == 0)
    real_as_fake = sum(1 for p, g in zip(predictions, ground_truths) if p == 0 and g == 1)

    apcer = fake_as_real / total_fake if total_fake > 0 else 0
    bpcer = real_as_fake / total_real if total_real > 0 else 0
    acer = (apcer + bpcer) / 2

    return {
        "total": len(image_paths),
        "proc": processed_images,
        "APCER": apcer,
        "BPCER": bpcer,
        "ACER": acer,
        "FPS": fps
    }


# ================== MAIN ==================
if __name__ == "__main__":
    datasets = {
        "CASIA-FASD": "./test/Casia-fasd/test_img/test_img/color",
        "CelebA-Spoof Mini": "./test/CelebA_Spoof-mini/val"
    }

    print("\n" + "=" * 60)
    print("BẮT ĐẦU ĐÁNH GIÁ PIPELINE FACE ANTI-SPOOFING FUSION")
    print("=" * 60)

    for name, path in datasets.items():
        if not os.path.exists(path):
            print(f"\n[!] Bỏ qua {name}: Không tìm thấy thư mục {path}")
            continue

        results = evaluate_dataset(path, name)

        # Hiển thị kết quả đúng định dạng f-string
        print(f"\n--- KẾT QUẢ: {name} ---")
        print(f"Tổng số tệp:       {results['total']}")
        print(f"Đã xử lý (có mặt): {results['proc']} ({(results['proc'] / results['total']) * 100:.2f}%)")
        print(f"APCER:             {results['APCER']:.4f}")
        print(f"BPCER:             {results['BPCER']:.4f}")
        print(f"ACER:              {results['ACER']:.4f}")
        print(f"Tốc độ trung bình: {results['FPS']:.2f} FPS")
        print("-" * 30)

    print("\n--- ĐÃ HOÀN TẤT ĐÁNH GIÁ TOÀN BỘ DATASET ---")