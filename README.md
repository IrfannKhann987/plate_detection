#🚗 Number Plate Recognition System

This project detects and reads vehicle number plates from images or video streams using YOLOv8, OpenCV, and EasyOCR.
It’s a complete pipeline — from object detection to text extraction — built as a practical computer vision application.

🧠 Overview

The system first detects number plates in an image using a fine-tuned YOLOv8 model trained on a Roboflow dataset.
After detection, the cropped plate region is passed through EasyOCR to extract the alphanumeric text.
This combination gives both speed (from YOLO) and accuracy (from OCR).

🏗️ Tech Stack

YOLOv8 – for number plate detection

OpenCV – for image processing and visualization

EasyOCR – for text extraction

Python – the core programming language

Roboflow – for dataset preparation and annotation

⚙️ How It Works

YOLOv8 Detection:

The model detects number plates and returns bounding box coordinates.

Region Cropping:

Each detected bounding box is cropped using OpenCV.

OCR Extraction:

EasyOCR reads the cropped region and extracts the plate text.

Display Results:

The final image displays bounding boxes with detected text above each plate.

📸 Results

Detects multiple plates in a single frame

Displays the plate number text clearly above each bounding box

Works on both images and real-time video streams
