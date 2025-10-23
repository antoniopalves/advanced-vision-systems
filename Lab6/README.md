# Feature Detection and Matching — AVS Lab 6

## Overview
This lab explores **corner detection, feature description, and matching** between images.  
It introduces both classical corner detectors and modern feature extraction techniques.

---

## Folder Contents
| File | Description |
|------|--------------|
| **6.1.py** | Harris corner detection on multiple image pairs. |
| **6.2.py** | Feature extraction, normalization, and patch-based matching. |
| **6.3.py** | FAST + Harris measure + BRIEF descriptor with custom Hamming matching. |
| **6.4.py** | ORB feature detection and matching for panoramic images. |
| **pm.py** | Utility functions for plotting and visualization of matches. |
| **fontanna1/2.jpg**, **budynek1/2.jpg**, **eiffel1/2.jpg** | Input image pairs for matching experiments. |

---

## 1. Harris Corner Detection
**File:** `6.1.py`  
Implements the **Harris Corner Detector** using Sobel gradients and Gaussian smoothing.

**Steps:**
1. Compute image gradients (Ix, Iy).  
2. Construct structure tensor components (Ixx, Iyy, Ixy).  
3. Compute Harris measure: `R = det(M) - k * (trace(M))^2`.  
4. Detect local maxima and threshold corners.  
5. Display detected corners on input images.

**Output:** red markers over corners for each image pair (Fontanna, Budynek).

---

## 2. Patch-Based Feature Matching
**File:** `6.2.py`  
Extends Harris detection with **patch-based descriptors** and **Euclidean-distance matching**.

**Pipeline:**
1. Detect corners using Harris response.  
2. Extract square patches around keypoints.  
3. Normalize intensity (zero mean, unit variance).  
4. Compare descriptors using Euclidean distance.  
5. Visualize top-N matches using `pm.plot_matches()`.

**Output:** feature match lines between image pairs.

---

## 3. FAST + Harris + BRIEF Descriptor
**File:** `6.3.py`  
Implements a hybrid detector and descriptor pipeline combining:
- **FAST** for rapid corner detection,  
- **Harris measure** for strength ranking,  
- **BRIEF** descriptor for binary comparison.

**Stages:**
1. FAST keypoint detection.  
2. Compute Harris measure and apply non-maximum suppression.  
3. Keep top N strongest corners.  
4. Compute intensity centroid orientation.  
5. Generate BRIEF descriptors and match using Hamming distance.

**Output:** Keypoints and best N matches between image pairs.

---

## 4. ORB Panorama Matching
**File:** `6.4.py`  
Uses **ORB (Oriented FAST and Rotated BRIEF)** to find matches between overlapping panoramas.

**Steps:**
1. Detect ORB keypoints.  
2. Compute binary descriptors.  
3. Match features with `cv2.BFMatcher(NORM_HAMMING)`.  
4. Sort and visualize the best matches.

**Output:** Match visualization between left and right panorama images.

---

## Dependencies
```bash
pip install numpy opencv-python matplotlib scipy
```

---

## Run Examples
```bash
python 6.1.py
python 6.2.py
python 6.3.py
python 6.4.py
```

Each script displays Harris corners, feature matches, or ORB correspondences.

---

**Author:** António Alves  
**Course:** Advanced Vision Systems (AVS) — FCT NOVA  
**Topic:** Feature Detection, Description, and Matching
