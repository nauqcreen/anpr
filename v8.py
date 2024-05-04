import math
import cv2
from ultralytics import YOLO


class ANPR_V8:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = YOLO(self.model_path)

    def detect(self, img, threshold=0.3):
        img = cv2.resize(img, (640, 640))
        results = self.model(img)
        plates = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                conf = math.ceil(box.conf[0] * 100) / 100
                if conf > threshold:
                    plates.append((x1, y1, x2, y2, conf))
        return plates, img
