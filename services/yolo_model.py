import os
import cv2
import numpy as np
from typing import List, Tuple
from ultralytics import YOLO
from dataclasses import dataclass

@dataclass
class Detection:
    bbox: Tuple[float, float, float, float]  # x, y, width, height
    confidence: float
    class_id: int
    class_name: str

class YOLOModel:
    def __init__(self, model_path: str, device: str = "cpu"):
        self.model_path = model_path
        self.device = device
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the YOLO model"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            self.model = YOLO(self.model_path)
            print(f"YOLO model loaded successfully from {self.model_path}")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            raise
    
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None
    
    async def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run YOLO detection on a frame"""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        try:
            # Run inference
            results = self.model(frame, device=self.device)
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        x, y, w, h = x1, y1, x2 - x1, y2 - y1
                        
                        # Get confidence and class
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = self.model.names[class_id]
                        
                        detection = Detection(
                            bbox=(x, y, w, h),
                            confidence=confidence,
                            class_id=class_id,
                            class_name=class_name
                        )
                        detections.append(detection)
            
            return detections
            
        except Exception as e:
            print(f"Error during YOLO detection: {e}")
            return []
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for YOLO input"""
        # Resize frame if needed
        height, width = frame.shape[:2]
        max_size = 640
        
        if max(height, width) > max_size:
            scale = max_size / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height))
        
        return frame
    
    def draw_detections(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Draw detection boxes on frame"""
        frame_copy = frame.copy()
        
        for detection in detections:
            x, y, w, h = detection.bbox
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
            
            # Draw bounding box
            color = (0, 255, 0) if detection.confidence > 0.7 else (0, 165, 255)
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{detection.class_name}: {detection.confidence:.2f}"
            cv2.putText(frame_copy, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame_copy 