# AVS Lab 2 – Foreground Moving Object Detection

This repository contains Python implementations of exercises 2.1 to 2.4 from **Laboratory 2** of the *Advanced Vision Systems* course (AGH University of Krakow). These tasks focus on foreground object detection through frame differencing and basic image analysis techniques.

## 📁 File Overview

| File       | Task Description                                        |
|------------|---------------------------------------------------------|
| Ex2.1.py   | Load and display image sequences (e.g., `pedestrians`)  |
| Ex2.2.py   | Perform frame differencing and binarization             |
| Ex2.3.py   | Label connected components and analyze the largest one  |
| Ex2.4.py   | Evaluate detection with precision, recall, and F1 score |

## 📦 Requirements

Make sure you have Python 3 installed along with the following libraries:
```bash
pip install opencv-python numpy

▶️ How to Run
Ensure the required video/image sequences (input/, groundtruth/, ROI, temporalROI.txt) are in the correct directories as expected by the scripts.

Run a script like this:

▶️ How to Run
Ensure the required video/image sequences (input/, groundtruth/, ROI, temporalROI.txt) are in the correct directories as expected by the scripts.

Run a script like this:
📚 Learning Objectives
Load and process video/image sequences using OpenCV

Apply frame subtraction and binarization

Use morphological operations to clean up binary masks

Label connected components and extract object statistics

Calculate precision, recall, and F1 score using reference ground truth

🎓 Course
Advanced Vision Systems
AGH University of Krakow
Lecturers: Dr. Tomasz Kryjak, MSc Mateusz Wąsala