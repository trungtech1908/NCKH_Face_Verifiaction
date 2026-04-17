import torch
import cv2
import os
import numpy as np
from collections import OrderedDict
from src.model_lib.MiniFASNet import MiniFASNetV2, MiniFASNetV1SE

class AntiSpoof:
    def __init__(self, model_dir):
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
        # 1. Chuyển sang RGB (Rất quan trọng cho FAS)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_res = cv2.resize(img_rgb, (80, 80))
        
        # 2. Chuyển sang Float và CHW
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
            
            # Dùng Softmax để lấy xác suất chính xác
            p27 = torch.softmax(out_27, dim=1)
            p40 = torch.softmax(out_40, dim=1)
            
            # Kết hợp xác suất
            combined = (p27 + p40) / 2
            # Prefer returning the probability of the "live" class (index 1) as the score
            # and a label that indicates live (1) only when real_prob >= 0.5.
            real_prob = combined[0, 1].item() if combined.shape[1] > 1 else float(combined[0].max())
            label = 1 if real_prob >= 0.5 else int(torch.argmax(combined, 1).item())
            return label, real_prob