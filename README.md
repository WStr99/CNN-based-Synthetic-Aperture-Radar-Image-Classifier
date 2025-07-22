# CNN-based-Synthetic-Aperture-Radar-image-classifier
This project focuses on detecting and classifying Soviet military vehicles in Synthetic Aperture Radar (SAR) images using the YOLOv8 object detection framework. The data originates from U.S. military satellite-based SAR systems and is annotated in PASCAL VOC format, which is converted to YOLO format for training.

The MSTAR dataset is a well-established benchmark for SAR-based target recognition, consisting of high-resolution radar images of ground vehicles under various configurations, depression angles, and conditions.
https://www.sdms.afrl.af.mil/index.php?collection=mstar 

# Labels
![val_batch0_labels](https://github.com/user-attachments/assets/4769146c-8815-43fd-ba3b-89a7912a2840)
![val_batch0_pred](https://github.com/user-attachments/assets/339385af-8335-4f29-9ab5-245fec568df7)

# Predictions
![val_batch2_labels](https://github.com/user-attachments/assets/be4b5b50-4396-4706-9d64-6f543f8a259b)
![val_batch2_pred](https://github.com/user-attachments/assets/3d78848f-d002-4617-8722-9265c6e310c0)

# Project Overview
- Dataset: MSTAR (Moving and Stationary Target Acquisition and Recognition) SAR imagery
- Classes: 10 Soviet military vehicle types
- Framework: Ultralytics YOLOv8
- Task: Object detection and classification
- Annotation Format: PASCAL VOC, converted to YOLO

# Model Performance
F1-Confidence Curve
The model achieves its optimal mean F1-score of 0.92 at a confidence threshold of 0.474, indicating high-quality detection across most classes. Each class's individual F1 trajectory shows strong confidence separation, especially above 0.3 confidence levels.
<img width="2250" height="1500" alt="F1_curve" src="https://github.com/user-attachments/assets/117cdbcf-4a6d-43bf-a3ec-4fd185602578" />

# Confusion Matrix (Normalized)
The normalized confusion matrix shows that the model performs well across most classes, with near-perfect classification in many cases. Notable observations:
- Class 0, 2, 3, 5, and 6 achieve ~100% accuracy
- Class 1 exhibits confusion with classes 9 and 7
- Background false positives are minimal
<img width="3000" height="2250" alt="confusion_matrix_normalized" src="https://github.com/user-attachments/assets/d0486ec9-f34a-40b9-8a8e-aa63f794a6c7" />

# Workflow
1. Annotation Conversion
The provided voc_to_yolo.py script converts VOC XML annotations to YOLOv8-compatible TXT files and splits the data into train/val/test sets.

Edit paths as needed:
xml_dir = r"path/to/Annotations"
img_dir = r"path/to/JPEGImages"
output_dir = r"path/to/output/yolo_dataset"

Run:
python voc_to_yolo.py

# Yolov8 Training
path: /absolute/path/to/yolo_dataset
train: images/train
val: images/val
test: images/test

names:
  0: tank
  1: apc
  2: truck
  3: missile_launcher
  4: radar
  5: transporter
  6: artillery
  7: armored_car
  8: command_vehicle
  9: launcher

yolo detect train model=yolov8s.pt data=data.yaml epochs=100 imgsz=640

# Applications
- Military SAR surveillance automation
- Target recognition in low-visibility or denied environments
- Real-time battlefield awareness tools
- Synthetic data evaluation for defense-related research

