
# Advanced Vision Systems - Lab: Visual Odometry (VO)

**Objective:** Estimate camera motion using visual information only (monocular visual odometry).

---

## Concept Summary

Visual Odometry (VO) is the process of estimating a robot or camera's motion (translation and rotation) using sequences of images captured from its environment. It is a specific case of **Structure from Motion (SfM)**, but optimized for **real-time pose estimation** instead of full 3D reconstruction.

---

## VO vs. SLAM

| **Aspect** | **Visual Odometry (VO)** | **Simultaneous Localization and Mapping (SLAM)** |
|-------------|---------------------------|--------------------------------------------------|
| Goal | Estimate motion trajectory | Estimate trajectory **and** reconstruct a consistent map |
| Drift | Accumulates over time | Minimized through loop closure and global optimization |
| Computation | Lighter, real-time | Heavier, includes map correction |
| Typical Output | Camera poses | Camera poses + environment map |

---

## Main Pipeline

1. **Feature Detection**  
   Identify distinct points in the image (commonly using ORB, SIFT, or FAST).  

2. **Feature Matching / Tracking**  
   Match keypoints between consecutive frames (e.g., Brute Force or FLANN matcher).  

3. **Triangulation**  
   Estimate 3D positions of matched points based on stereo geometry or motion parallax.  

4. **RANSAC and Inlier Filtering**  
   Filter noisy correspondences to retain geometrically consistent matches.  

5. **Pose Estimation**  
   Recover camera rotation **R** and translation **t** using methods like `cv2.findEssentialMat` and `cv2.recoverPose`.

---

## Code Summary (`[student]_vo_en.py`)

The provided notebook implements a **monocular VO system** with OpenCV functions for:
- Feature extraction (ORB).
- Matching keypoints between consecutive frames.
- Estimating the essential matrix for camera motion recovery.
- Reconstructing the trajectory incrementally frame by frame.
- Optional visualization of the estimated path.

---

## Dependencies

```bash
pip install numpy opencv-python matplotlib
```

---

## Example Run

```bash
python [student]_vo_en.py
```

Output includes:
- Visualized camera path.
- Console logs of rotation and translation vectors.

---

## Notes

- The dataset must contain **sequential images** (monocular or stereo).  
- Accuracy depends on lighting conditions, feature richness, and motion smoothness.  
- Drift correction is not implemented (for that, SLAM would be required).

---

## Author
António Alves — Erasmus Advanced Vision Systems Lab, FCT NOVA
