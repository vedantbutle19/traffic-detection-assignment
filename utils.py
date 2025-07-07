import cv2
import supervision as sv
from collections import Counter

# Mapping COCO class IDs to vehicle names
vehicle_classes = {
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck'
}

def summarize_and_annotate(image, detections):
    # Prepare labels with class names and confidence scores
    labels = [
        f"{vehicle_classes[cls_id]}: {conf:.2f}"
        for cls_id, conf in zip(detections.class_id, detections.confidence)
    ]

    # Count vehicle types
    vehicle_counts = Counter()
    for cls_id in detections.class_id:
        if cls_id in vehicle_classes:
            vehicle_counts[vehicle_classes[cls_id]] += 1

    # Draw bounding boxes
    box_annotator = sv.BoxAnnotator(thickness=2)
    annotated_image = box_annotator.annotate(image, detections)

    # Draw labels
    label_annotator = sv.LabelAnnotator(text_scale=0.7, text_thickness=1, text_padding=3)
    annotated_image = label_annotator.annotate(annotated_image, detections, labels=labels)

    # Prepare summary text
    summary_text = f"Vehicle Counts: {dict(vehicle_counts)}"
    font_scale = 0.8
    text_size, _ = cv2.getTextSize(summary_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
    text_w, text_h = text_size

    # Semi-transparent background
    overlay = annotated_image.copy()
    cv2.rectangle(
        overlay,
        (10, annotated_image.shape[0] - text_h - 20),
        (10 + text_w + 10, annotated_image.shape[0] - 10),
        (0, 0, 0),
        -1
    )
    alpha = 0.5
    cv2.addWeighted(overlay, alpha, annotated_image, 1 - alpha, 0, annotated_image)

    # Draw the summary text
    cv2.putText(
        img=annotated_image,
        text=summary_text,
        org=(15, annotated_image.shape[0] - 15),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=font_scale,
        color=(255, 255, 255),
        thickness=2,
        lineType=cv2.LINE_AA
    )

    return annotated_image
