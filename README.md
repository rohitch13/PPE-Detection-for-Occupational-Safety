# PPE Detection on Construction Site Workers using YOLOv8

This project focuses on detecting Personal Protective Equipment (PPE) on construction site workers using computer vision techniques with the YOLOv8 object detection framework. It includes data preprocessing, augmentation, model training, evaluation, and deployment-ready insights.

---

## Project Objective

To build a robust and real-time object detection model capable of identifying PPE compliance (e.g., hardhats, masks, vests) in construction environments using YOLOv8.

---

## Dataset Overview

- **Source**: https://www.kaggle.com/datasets/snehilsanyal/construction-site-safety-image-dataset-roboflow
- **Annotations**: Bounding boxes for 10 classes including:
  - Hardhat, Mask, NO-Hardhat, NO-Mask, NO-Safety Vest, Person, Safety Cone, Safety Vest, Machinery, Vehicle.
- **Distribution After Structuring**:
  - `Train`: 2,241 images (80%)
  - `Validation`: 420 images (14%)
  - `Test`: 140 images (5%)

---

## Data Assumptions

- Each image may contain multiple annotations from the same or different classes.
- All annotations are assumed to be in YOLO format with normalized coordinates.
- Data split is stratified to ensure class representation across train/val/test sets.

---

## Preprocessing & Augmentation

- Data structuring with proper directory and label alignment.
- Augmentation techniques:
  - Horizontal flipping
  - Random rotation
  - Brightness/contrast adjustment
- Annotation consistency maintained post-transformation.

---

## Model Configuration

- **Base Model**: YOLOv8n/YOLOv11n (`ultralytics`)
- **Training Parameters**:
  - Epochs: 200
  - Image Size: 640x640
  - Batch Size: 16
  - Device: GPU (if available)
- **Metrics Captured**: Precision, Recall, mAP@0.5, mAP@0.5:0.95, F1-Score

---

## Inference Example

![image](https://github.com/user-attachments/assets/0f27c054-0e5f-4bef-98de-e3bf4390bc80)

```python

