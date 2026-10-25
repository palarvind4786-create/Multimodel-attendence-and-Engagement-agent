import cv2
import numpy as np
from typing import List, Tuple

class PersonDetector:
    def __init__(self, confidence_threshold: float = 0.45):
        self.confidence_threshold = confidence_threshold
        self.yolo_model = None
        self.use_fallback = False
        self.device = "cpu"
        
        try:
            import torch
            from ultralytics import YOLO
            self.device = 0 if torch.cuda.is_available() else "cpu"
            self.yolo_model = YOLO("yolov8n.pt")
            # Warm up
            _ = self.yolo_model(np.zeros((320, 320, 3), dtype=np.uint8), device=self.device, verbose=False)
        except Exception as e:
            print(f"[PersonDetector] YOLOv8 init warning ({e}). Using OpenCV HOG fallback.")
            self.use_fallback = True
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Detects persons in the frame.
        Returns list of bounding boxes: (x1, y1, x2, y2, confidence)
        """
        if frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]
        boxes = []

        if not self.use_fallback and self.yolo_model is not None:
            try:
                results = self.yolo_model(
                    frame,
                    classes=[0],
                    conf=self.confidence_threshold,
                    device=self.device,
                    verbose=False
                )
                for r in results:
                    for box in r.boxes:
                        coords = box.xyxy[0].cpu().numpy().astype(int)
                        conf = float(box.conf[0].cpu().numpy())
                        x1 = max(0, min(w - 1, int(coords[0])))
                        y1 = max(0, min(h - 1, int(coords[1])))
                        x2 = max(0, min(w - 1, int(coords[2])))
                        y2 = max(0, min(h - 1, int(coords[3])))
                        if x2 > x1 and y2 > y1:
                            boxes.append((x1, y1, x2, y2, conf))
                return boxes
            except Exception as e:
                print(f"[PersonDetector] Inference error on {self.device}: {e}, attempting CPU/HOG fallback.")
                if self.device != "cpu":
                    self.device = "cpu"

        # Fallback HOG person detector
        if hasattr(self, 'hog'):
            rects, weights = self.hog.detectMultiScale(
                frame,
                winStride=(8, 8),
                padding=(4, 4),
                scale=1.05
            )
            for (x, y, bw, bh), conf in zip(rects, weights):
                score = float(conf[0]) if hasattr(conf, '__getitem__') else float(conf)
                boxes.append((x, y, x + bw, y + bh, max(0.5, min(0.95, score))))

        return boxes
