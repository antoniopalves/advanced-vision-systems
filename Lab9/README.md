
# Advanced Vision Systems - Lab 9
**Topic:** Generalized Hough Transform (GHT) — Shape Detection

---

## Overview
This lab implements the **Generalized Hough Transform**, a robust image processing method used to detect complex shapes in images — independent of their position, rotation, or partial occlusion.  
While the classical Hough Transform detects simple geometric primitives (lines, circles, ellipses), GHT extends this idea to arbitrary templates by using a lookup table of gradient-based offsets (the **R-table**).

---

## Method Summary

1. **Template Extraction (`trybik.jpg`)**  
   - Load and binarize the gear pattern.  
   - Detect the main contour.  
   - Compute gradient orientation and magnitude using the Sobel operator.  
   - Compute the object’s **center of gravity** and create an **R-table** linking gradient angles to radial offsets.

2. **Target Detection (`trybiki2.jpg`)**  
   - Compute gradients of the target image.  
   - For each edge pixel, vote in a Hough accumulator space using the R-table.  
   - Peaks in this accumulator correspond to likely object centers.

3. **Post-Processing**  
   - Apply a **maximum filter** to isolate significant peaks.  
   - Mark detected gears in red and overlay contours on the original image.  
   - Display the **Hough space** and the final detection image.

---

## Files

| File | Description |
|------|--------------|
| `9.1.py` | Main script implementing the GHT algorithm |
| `trybik.jpg` | Reference pattern (gear template) |
| `trybiki2.jpg` | Target image with multiple gears |

---

## Dependencies

```bash
pip install numpy opencv-python matplotlib scipy
```

---

## Execution

```bash
python 9.1.py
```

**Output:**
- A window with detected gear contours (red).  
- A plot showing the Hough space (accumulator peaks).

---

## Parameters

| Parameter | Role | Default |
|------------|------|----------|
| `num_patterns_to_detect` | Number of shapes to detect | 5 |
| `footprint` | Neighborhood size for maximum filter | 40×40 |
| `gradient_magnitude_t > 0.5` | Edge threshold | — |

---

## Author
António Alves — Erasmus Advanced Vision Systems Lab, FCT NOVA
