
# Advanced Vision Systems - Lab 13
**Topic:** Event-Based Vision and Dynamic Vision Sensors (DVS)

---

## Overview
This lab explores **event-based image acquisition** using **Dynamic Vision Sensors (DVS)** — neuromorphic sensors that asynchronously record brightness changes in the scene instead of capturing full image frames at fixed rates.  
Each “event” encodes the **pixel position (x, y)**, **timestamp (t)**, and **polarity (p)** indicating whether brightness increased or decreased.

The exercises demonstrate:
- Reading and visualizing DVS event data in 3D space.
- Temporal segmentation of event streams into frames.
- Understanding temporal resolution, polarity, and object motion representation.

---

## Concepts Recap

| Concept | Description |
|----------|--------------|
| **Event** | A brightness change detected at a pixel, encoded as (t, x, y, p). |
| **Polarity (p)** | +1 for intensity increase, –1 for decrease. |
| **Asynchronous capture** | Pixels trigger independently; no fixed frame rate. |
| **Temporal resolution** | Microsecond-level timing precision. |
| **Event frame integration** | Accumulating events within a time window (τ) to form images. |

---

## Files

| File | Description |
|------|--------------|
| `13.1.py` | Reads and visualizes event data in 3D (x, y, time). |
| `13.2.py` | Filters events by time range, visualizes 3D polarity distributions, and analyses timing properties. |
| `13.3.py` | Generates event-based image frames with different integration windows (τ) to study temporal aggregation. |

---

## Pipeline Summary

### 1. Data Parsing (`13.1.py`)
- Reads `events.txt`, filters timestamps < 1 second.
- Computes number of positive/negative events.
- Visualizes them as a **3D scatter plot** (red = positive, blue = negative).

### 2. Temporal Filtering & Analysis (`13.2.py`)
- Reads subsets of events (first 8000, or within 0.5–1.0 s range).
- Plots 3D trajectories to analyze motion direction and event polarity.
- Answers key questions:
  - Sequence duration in seconds.
  - Event timestamp resolution (≈ 1 μs).
  - Dependence of time differences on scene motion.

### 3. Event Frame Generation (`13.3.py`)
- Aggregates events over a temporal window τ:
  - **τ = 0.001 s:** high temporal detail, noisy.
  - **τ = 0.01 s:** balanced, clear motion contours.
  - **τ = 0.1 s:** smooth but temporally coarse.
- Displays reconstructed binary event frames.

---

## Dependencies

```bash
pip install numpy matplotlib opencv-python
```

---

## Execution

### Visualize and Analyze Events
```bash
python 13.1.py
python 13.2.py
```

### Generate Event Frames
```bash
python 13.3.py
```

Press any key to progress through event frames.

---

## Outputs

- **3D scatter plots** showing event spatiotemporal distributions.  
- **Binary event frames** visualizing movement across time windows.  
- **Quantitative analysis** of event rates, polarities, and motion direction.

---

## Author
António Alves — Erasmus Advanced Vision Systems Lab, FCT NOVA
