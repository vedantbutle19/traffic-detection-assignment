# traffic-detection-assignment
Vehicle Detection and Counting
This project uses the YOLOv8 model to detect and count vehicles in an image, annotating the results with bounding boxes, class labels, confidence scores, and a summary of vehicle counts.
Table of Contents

Overview
Requirements
Installation
Usage
Code Explanation
Output
License

Overview
The script processes an input image to detect vehicles (cars, motorcycles, buses, trucks) using the YOLOv8 medium model. It applies non-maximum suppression (NMS) to filter overlapping detections, annotates the image with bounding boxes and labels, and displays a summary of vehicle counts overlaid on the image.
Requirements

Python 3.8+
Libraries:
supervision (for detection handling and visualization)
inference (for YOLOv8 model inference)
opencv-python (for image processing)
numpy (for numerical operations)
collections (for counting vehicles)

Installation

Clone or download this repository.
Install the required dependencies:pip install supervision inference opencv-python numpy

Ensure you have internet access to download the input image and model weights.

Usage

Run the script:python vehicle_detection.py

The script will:
Load an image from the specified URL.
Perform vehicle detection using YOLOv8m-1280.
Annotate the image with bounding boxes, class labels, and confidence scores.
Display the vehicle count summary on the image.
Show the annotated image.

Code Explanation

Image Loading: Loads an image in BGR format from a URL using load_image_bgr.
Model Inference: Uses YOLOv8 medium model (yolov8m-1280) with a confidence threshold of 0.3.
Non-Maximum Suppression: Applies NMS with a threshold of 0.4 to reduce overlapping detections.
Vehicle Filtering: Filters detections to include only vehicles (car, motorcycle, bus, truck) based on COCO class IDs.
Annotation: Draws bounding boxes and labels (class name + confidence) using supervision's BoxAnnotator and LabelAnnotator.
Vehicle Counting: Counts vehicles by type using Counter.
Summary Overlay: Adds a semi-transparent background and text summary of vehicle counts using OpenCV.
Output: Displays the annotated image using supervision's plot_image.

Output
The script outputs an annotated image with:

Bounding boxes around detected vehicles.
Labels showing vehicle type and confidence score.
A text summary at the bottom showing the count of each vehicle type (e.g., Vehicle Counts: {'car': 5, 'truck': 2}).

License
This project is licensed under the MIT License.
