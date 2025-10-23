
# Advanced Vision Systems - Lab 11
**Topic:** Pedestrian Detection using HOG (Histogram of Oriented Gradients) + SVM

---

## Overview
This lab implements the classical **HOG + SVM** pipeline for pedestrian detection — a foundational method in computer vision before the rise of deep learning detectors.  
It focuses on computing **gradient-based features (HOG)** and training a **Support Vector Machine (SVM)** classifier to distinguish between human and non-human image windows.

---

## Method Summary

### 1. Gradient Computation (11.1.py)
- Compute **horizontal** and **vertical gradients** using the **Sobel operator**.
- Derive **magnitude** and **orientation** of gradients.
- Quantize orientations into bins and construct **cell histograms (8×8 px)**.
- Normalize histograms over **blocks (2×2 cells)** to obtain illumination-invariant feature vectors.
- Visualize histograms per orientation bin.

### 2. Feature Extraction and Training (11.2.py / 11.3.py)
- Compute OpenCV’s HOG descriptors for all samples in the dataset:
  - **Positive samples:** cropped pedestrian images.  
  - **Negative samples:** random background images.
- Train a **linear SVM** classifier to separate both classes.
- Evaluate performance using accuracy and confusion matrix.
- Save the trained model to `svm_pedestrian_detector.pkl`.

### 3. Detection Stage
- Slide a detection window (64×128 px) over the test image at multiple scales.
- For each window, compute its HOG descriptor and classify it with the SVM.
- Mark detected pedestrian regions with green bounding boxes.

---

## Files

| File | Description |
|------|--------------|
| `11.1.py` | Manual implementation of HOG descriptor computation |
| `11.2.py` | SVM training and evaluation using HOG features |
| `11.3.py` | Final pedestrian detector using trained SVM model |
| `svm_pedestrian_detector.pkl` | Trained SVM model |
| `testImage1.png`–`testImage4.png` | Test images for visual validation |

---

## HOG Parameters

| Parameter | Description | Value |
|------------|--------------|--------|
| Window size | Detection region | (64, 128) |
| Block size | Normalization block | (16, 16) |
| Block stride | Block step | (8, 8) |
| Cell size | Histogram cell | (8, 8) |
| Bins | Orientation bins | 9 |

---

## Dependencies

```bash
pip install numpy opencv-python matplotlib scipy scikit-learn joblib
```

---

## Execution

### Step 1: Compute and visualize HOG
```bash
python 11.1.py
```

### Step 2: Train SVM
```bash
python 11.2.py
```

### Step 3: Run detection
```bash
python 11.3.py
```

Press **Q** to close visualization windows.

---

## Output

- **HOG histograms:** Visual display of gradient orientations.  
- **SVM training report:** Accuracy and confusion matrix on training set.  
- **Detection images:** Bounding boxes marking detected pedestrians.

---

## Author
António Alves — Erasmus Advanced Vision Systems Lab, FCT NOVA
