
# Advanced Vision Systems - Lab 7
**Topic:** Discriminative Correlation Filter (DCF) Object Tracking

## Overview
This lab implements a visual tracking algorithm based on **Discriminative Correlation Filters (DCF)**. The method estimates an object's motion across frames using correlation in the **frequency domain** — ensuring **low computational cost** and **real-time feasibility**.

The algorithm is based on the paper:  
*Bolme, D. S., Beveridge, J. R., Draper, B. A., & Lui, Y. M. (2010). Visual Object Tracking using Adaptive Correlation Filters. CVPR.*  
[Link](https://www.cs.colostate.edu/~draper/papers/bolme_cvpr10.pdf)

## Core Idea
The tracker maintains a learned correlation filter that represents the tracked object. For each new frame, it performs correlation between the learned filter and a search region around the previous object position, determining where the filter response peaks.

### Steps
1. **Initialization**:  
   - The target is selected in the first frame.  
   - A Gaussian response map is generated around the target center.  
   - The initial filter is computed in the frequency domain.

2. **Pre‑Training**:  
   - The target patch is randomly warped to increase robustness.  
   - Frequency domain representations are averaged over multiple warped samples.

3. **Tracking Loop**:  
   - The region around the last known position is extracted.  
   - Correlation response is computed to find the new location.  
   - The filter is updated gradually using a learning rate.

4. **Evaluation**:  
   - Overlaps between predictions and ground truth boxes are computed via IoU.  
   - Mean IoU is reported per sequence.

## Files
- `7.1.py`: Main DCF tracking implementation (OpenCV + NumPy).
- `Antonio_Alves_DCF.py`: Extended version including algorithm description and Colab-compatible notebook code.

## Parameters
| Variable | Description | Default |
|-----------|--------------|----------|
| `SIGMA` | Gaussian standard deviation for response map | 17 |
| `SEARCH_REGION_SCALE` | Search window expansion factor | 2 |
| `LR` | Learning rate for filter update | 0.125 |
| `NUM_PRETRAIN` | Number of pre-training augmentations | 128 |
| `VISUALIZE` | Show results with OpenCV windows | True |

## How to Run
1. Set the dataset path in `DATASET_DIR` (e.g. `Lab7/sequences`).  
2. Place each sequence folder with subfolders `color/` and file `groundtruth.txt`.  
3. Run the script:
   ```bash
   python 7.1.py
   ```
4. Press **Q** to exit visualization windows.

## Output
- Tracker visualizations (bounding boxes on each frame).
- Mean IoU value per sequence in terminal.

## Dependencies
```bash
pip install numpy opencv-python imutils
```

## Author
António Alves — Erasmus Advanced Vision Systems Lab, FCT NOVA
