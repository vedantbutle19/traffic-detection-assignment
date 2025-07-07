import supervision as sv
from inference import get_model
from inference.core.utils.image_utils import load_image_bgr
import numpy as np
from utils import vehicle_classes

class VehicleDetector:
    def __init__(self, confidence=0.3, nms_threshold=0.4):
        self.model = get_model(model_id="yolov8m-1280")
        self.confidence = confidence
        self.nms_threshold = nms_threshold

    def detect_vehicles(self, image_url):
        # Load and infer
        image = load_image_bgr(image_url)
        results = self.model.infer(image, confidence=self.confidence)[0]
        detections = sv.Detections.from_inference(results)

        # Apply non-max suppression
        detections = detections.with_nms(threshold=self.nms_threshold)

        # Filter only vehicles
        vehicle_mask = np.isin(detections.class_id, list(vehicle_classes.keys()))
        detections = detections[vehicle_mask]

        return image, detections
