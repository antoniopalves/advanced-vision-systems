# Optical Flow — AVS Lab 4

## Overview
This lab explores **optical flow estimation** using both **classical** and **deep learning–based** approaches.  
It aims to understand how motion between consecutive frames can be estimated through pixel correspondences and learned flow fields.

---

## Folder Contents
| File | Description |
|------|--------------|
| **4.1.py** | Block-based optical flow (Sum of Squared Differences). |
| **4.2.py** | Multi-scale (pyramidal) block matching. |
| **correlation.py** | CUDA kernel for correlation layer (used in LiteFlowNet). |
| **run_spynet.py** | Deep learning optical flow using **SpyNet** (PyTorch). |
| **run_liteflownet.py** | Deep learning optical flow using **LiteFlowNet** (PyTorch). |
| **I.jpg, J.jpg** | Input frame pair used for classical methods. |
| **cm1.png, cm2.png, cm_gt.png** | Color maps of flow results and ground truth. |

---

## Classical Methods

### **4.1.py — Block Matching**
Implements dense optical flow by comparing square blocks between two grayscale images.  
For each pixel, the algorithm finds the block displacement that minimizes the **Sum of Squared Differences (SSD)**.

**Steps:**
1. Convert frames to grayscale.  
2. Define search and block windows.  
3. Compute SSD for every displacement.  
4. Assign the best (dx, dy) as the local flow vector.  
5. Visualize results in HSV (Hue: direction, Saturation: magnitude).

**Output:** HSV image showing motion direction and intensity.

---

### **4.2.py — Image Pyramid Approach**
Extends the block matching method with a **multi-scale pyramid**:
- Coarser resolutions capture large displacements.
- Finer levels refine local details.

**Pipeline:**
1. Build Gaussian pyramids for both images.  
2. Compute optical flow at each scale.  
3. Upsample flow to higher resolution and refine iteratively.  
4. Visualize results per scale using HSV encoding.

---

## Deep Learning Methods

### **run_spynet.py — SpyNet**
Implementation of **SpyNet**, a compact CNN trained for optical flow estimation.  
SpyNet combines:
- Pyramid processing (coarse-to-fine).
- Small convolutional refinement networks at each level.

**Features:**
- Fast inference on GPU.  
- Lightweight (~1.2M parameters).  
- Pretrained on **Sintel** dataset.  
- Outputs dense flow maps for arbitrary image pairs.

**Visualization:** Uses OpenCV HSV color coding for direction and magnitude.

---

### **run_liteflownet.py — LiteFlowNet**
A high-performance optical flow model based on **correlation layers** and **feature warping**.  
Implements the complete LiteFlowNet pipeline:
- Feature extraction.  
- Cost volume computation (via `correlation.py`).  
- Multi-stage matching and refinement.  
- Sub-pixel flow correction and regularization.

**Requirements:**
- CUDA + cuDNN  
- PyTorch ≥ 1.3  
- Internet access for pretrained model download

**Output:** Flow visualization per frame pair using HSV mapping.

---

## Comparison
| Method | Type | Data Requirement | Speed | Accuracy | GPU Use |
|--------|------|------------------|--------|-----------|----------|
| 4.1.py | Classical SSD | None | Slow | Low | No |
| 4.2.py | Classical Pyramid | None | Moderate | Medium | No |
| SpyNet | CNN | Pretrained | Fast | High | Yes |
| LiteFlowNet | CNN | Pretrained | Medium | Very High | Yes |

---

## Dependencies
```bash
pip install numpy opencv-python torch torchvision pillow
```

For GPU-based models:
```bash
pip install cupy-cuda12x
```

---

## Run Examples

### Classical
```bash
python 4.1.py
python 4.2.py
```

### Deep Learning
```bash
python run_spynet.py
python run_liteflownet.py
```

Each will display a **flow visualization window** showing pixel motion between frames.

---

**Author:** António Alves  
**Course:** Advanced Vision Systems (AVS) — FCT NOVA  
**Topic:** Optical Flow — Classical vs Deep Learning Methods
