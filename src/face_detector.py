import sys
import os
import torch
import cv2
import numpy as np

RETINAFACE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../external/retinaface")
)
sys.path.insert(0, RETINAFACE_PATH)

from models.retinaface import RetinaFace
from data import cfg_mnet
from layers.functions.prior_box import PriorBox
from utils.box_utils import decode
from utils.nms.py_cpu_nms import py_cpu_nms


class FaceDetector:

    def __init__(self, weight_path):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = RetinaFace(cfg=cfg_mnet, phase='test')
        self.model.load_state_dict(torch.load(weight_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()

        self.confidence_threshold = 0.6
        self.nms_threshold = 0.4

        print("FaceDetector loaded")

    def detect(self, img):

        img = np.float32(img)

        im_height, im_width, _ = img.shape
        scale = torch.Tensor([im_width, im_height, im_width, im_height])

        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0).to(self.device)

        scale = scale.to(self.device)

        with torch.no_grad():
            loc, conf, _ = self.model(img)

        priorbox = PriorBox(cfg_mnet, image_size=(im_height, im_width))
        priors = priorbox.forward().to(self.device)

        boxes = decode(loc.data.squeeze(0), priors, cfg_mnet['variance'])
        boxes = boxes * scale
        boxes = boxes.cpu().numpy()

        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

        inds = np.where(scores > self.confidence_threshold)[0]

        boxes = boxes[inds]
        scores = scores[inds]

        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32)

        keep = py_cpu_nms(dets, self.nms_threshold)
        dets = dets[keep]

        results = []

        for d in dets:
            x1, y1, x2, y2, score = d

            results.append({
            "box": [int(x1), int(y1), int(x2), int(y2)],
            "score": float(score)
        })

        return results
    def detect_and_crop(self, img, size=112):

        detections = self.detect(img)

        faces = []

        for det in detections:

            x1, y1, x2, y2 = det["box"]

            face = img[y1:y2, x1:x2]

            if face.size == 0:
                continue

            face = cv2.resize(face, (size, size))

            faces.append({
            "box": det["box"],
            "score": det["score"],
            "face": face
        })

        return faces