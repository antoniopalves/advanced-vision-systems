# Camera Calibration and Stereo Vision — AVS Lab 5

## Overview
This lab focuses on **camera calibration**, **fisheye distortion correction**, and **stereo vision** using OpenCV.  
The goal is to estimate intrinsic and extrinsic camera parameters, rectify stereo image pairs, and compute disparity maps.

---

## Folder Contents
| File | Description |
|------|--------------|
| **single_camera_calibration.py** | Calibrates a single fisheye camera using multiple chessboard images. |
| **stereo_camera_calibration.py** | Performs stereo calibration and rectification between left/right cameras. |
| **5.3.py** | Computes disparity maps after calibration using StereoBM. |

---

## 1. Single Camera Calibration
**File:** `single_camera_calibration.py`  
Calibrates one camera using multiple chessboard images to estimate:
- Intrinsic matrix `K`
- Distortion coefficients `D`

**Steps:**
1. Detect chessboard corners in all calibration images.  
2. Refine corners to sub-pixel precision.  
3. Compute calibration matrices with `cv2.fisheye.calibrate()`.  
4. Generate rectification maps using `cv2.fisheye.initUndistortRectifyMap()`.  
5. Display undistorted images for visual comparison.

**Output:**  
- `K`: Intrinsic camera matrix  
- `D`: Fisheye distortion coefficients  
- Visualization of original vs. undistorted image

---

## 2. Stereo Camera Calibration
**File:** `stereo_camera_calibration.py`  
Estimates the geometric relationship between two calibrated fisheye cameras.

**Pipeline:**
1. Detect chessboard corners in both left and right images.  
2. Calibrate each camera individually (`cv2.fisheye.calibrate`).  
3. Run `cv2.fisheye.stereoCalibrate()` to get:  
   - Rotation matrix (R)  
   - Translation vector (T)  
4. Rectify both views using `cv2.fisheye.stereoRectify()`.  
5. Generate rectification maps and visualize aligned images with horizontal lines.

**Output:**  
Side-by-side rectified stereo image with horizontal alignment lines.

---

## 3. Stereo Disparity Computation
**File:** `5.3.py`  
Uses the calibration parameters to compute depth (disparity) between left/right views.

**Steps:**
1. Calibrate both cameras using chessboard pairs.  
2. Undistort and rectify stereo images.  
3. Compute disparity maps using `cv2.StereoBM_create()`.  
4. Display left/right rectified images and disparity maps.

**Output:**  
- Disparity map (visual representation of depth)  
- Rectified left/right image pairs

---

## Dependencies
```bash
pip install numpy opencv-python
```

Optional (for visualization):
```bash
pip install matplotlib
```

---

## Run Examples
```bash
python single_camera_calibration.py
python stereo_camera_calibration.py
python 5.3.py
```

---

**Author:** António Alves  
**Course:** Advanced Vision Systems (AVS) — FCT NOVA  
**Topic:** Camera Calibration and Stereo Depth Estimation
