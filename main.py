from detector import VehicleDetector
from utils import summarize_and_annotate
import supervision as sv

# Image URL
image_url = "https://media.ahmedabadmirror.com/am/uploads/mediaGallery/image/1651869427801.jpg-org"

# Initialize detector and run detection
detector = VehicleDetector(confidence=0.3, nms_threshold=0.4)
image, detections = detector.detect_vehicles(image_url)

# Annotate image and add summary
annotated_image = summarize_and_annotate(image, detections)

# Display final image
sv.plot_image(annotated_image)
