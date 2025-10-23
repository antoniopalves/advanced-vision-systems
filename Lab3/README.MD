# Background Subtraction Experiments — AVS Lab

## Overview
This repository contains Python scripts developed for the **Advanced Vision Systems (AVS)** course.  
Each script implements and evaluates a **background subtraction** method using image sequences from the **PETS Pedestrian Dataset**.  
Goal: compare traditional algorithms with deep-learning-based inference (BSUV-Net).

---

## Structure
- **3.1.py** → Mean/Median Buffer Method  
- **3.2.py** → Incremental Mean and Median Approximation  
- **3.3.py** → Conservative Update for Background Models  
- **3.4.py** → OpenCV MOG2 Background Subtraction  
- **3.5.py** → OpenCV KNN Background Subtraction  
- **inference.py** → BSUV-Net Deep Learning Inference  
- **infer_config_autoBG.py** → Config file for automatic background setup

---

## Scripts Summary

### 3.1 — Buffer-Based Background Modeling
Uses a sliding buffer of N frames to compute mean and median background models.  
Foreground is detected via absolute difference and thresholding.  
Outputs precision, recall, and F1-score for both mean and median models.

### 3.2 — Incremental Approximation (Mean & Median)
Implements recursive updates:
- Mean: `B_t = α·I_t + (1−α)·B_{t−1}`
- Median: pixel-wise incremental adjustment  
Memory efficient and adaptive to slow illumination changes.

### 3.3 — Conservative Update
Extends 3.2 by updating the background **only on background pixels**.  
Avoids contamination by moving objects.  
Computes precision, recall, and F1 for both mean and median methods.

### 3.4 — MOG2 (Gaussian Mixture Model)
Uses `cv2.createBackgroundSubtractorMOG2()`.  
Models complex backgrounds with multiple intensity modes per pixel.  
Provides better noise tolerance and adaptability.

### 3.5 — KNN Background Subtraction
Uses `cv2.createBackgroundSubtractorKNN()`.  
Pixel classification via K-nearest-neighbors statistics.  
Evaluated on the same pedestrian dataset for comparison.

### inference.py — BSUV-Net Deep Learning
Runs background subtraction using a pretrained **BSUV-Net** model (PyTorch).  
Loads a video, processes each frame, and saves visualization output combining:
- Original frame  
- Binary mask  
- Overlay (foreground on background)

Outputs FPS and segmentation results.

### infer_config_autoBG.py — Configuration
Holds parameters for inference:
- Model path  
- Background type (automatic/manual)  
- Normalization parameters  
- Optional HRNet semantic segmentation setup

---

## How to Run

### Classical Methods
```bash
python 3.1.py
python 3.2.py
python 3.3.py
python 3.4.py
python 3.5.py
```
Dataset paths inside each script:
- Input: `...\Lab2\pedestrian\input`
- Ground truth: `...\Lab2\pedestrian\groundtruth`

### Deep Learning Inference
```bash
python inference.py
```
Edit `inp_path` and `out_path` inside the script to match your files.

---

## Evaluation Metrics
All scripts report:
- **Precision (P)** = TP / (TP + FP)  
- **Recall (R)** = TP / (TP + FN)  
- **F1 Score** = 2PR / (P + R)

---

## Dependencies
```bash
pip install numpy opencv-python torch torchvision
```

---

## Summary Table

| Script | Method | Type | Adaptive | Deep Learning | Conservative Update |
|--------|---------|------|-----------|----------------|---------------------|
| 3.1 | Buffer Mean/Median | Classical | ✗ | ✗ | ✗ |
| 3.2 | Incremental Approximation | Classical | ✓ | ✗ | ✗ |
| 3.3 | Conservative Mean/Median | Classical | ✓ | ✗ | ✓ |
| 3.4 | MOG2 | OpenCV | ✓ | ✗ | ✗ |
| 3.5 | KNN | OpenCV | ✓ | ✗ | ✗ |
| inference.py | BSUV-Net | Deep Learning | ✓ | ✓ | ✓ |

---

**Author:** António Alves  
**Course:** Advanced Vision Systems (AVS), FCT NOVA  
**Purpose:** Comparative study of background subtraction methods in video surveillance
