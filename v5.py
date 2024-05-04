import cv2
import torch
import math


class ANPR_V5:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=self.model_path, force_reload=True)

    def detect(self, img, threshold=0.1):
        img = cv2.resize(img, (640, 640))
        results = self.model(img)
        plates = []
        for r in results.xyxy[0]:
            x1, y1, x2, y2 = int(r[0]), int(r[1]), int(r[2]), int(r[3])
            conf = math.ceil(r[4] * 100) / 100
            if conf > threshold:
                plates.append((x1, y1, x2, y2, conf))
        return plates, img
