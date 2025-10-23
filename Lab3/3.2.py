import cv2
import os
import numpy as np

# Set the path to the folder containing the image sequence
sequence_folder = r"c:\Users\anton\Documents\Erasmus\AVS\Lab2\pedestrian\input"  # change this to the appropriate folder

# Specify the start and end frame indices
start_frame = 300
end_frame = 1100

# Specify the step for frame analysis
step = 1  # Adjust this value as needed

# Initialize counters for evaluation metrics for mean approximation
TP_mean = 0  # True Positive
FP_mean = 0  # False Positive
FN_mean = 0  # False Negative

# Initialize counters for evaluation metrics for median approximation
TP_median = 0
FP_median = 0
FN_median = 0

# Parameters for mean approximation method
alpha = 0.01

# Buffer initialization for mean approximation method
BUF_mean = None

# Parameters for median approximation method
BGN_median = None

# Main loop
for i in range(start_frame + 1, end_frame, step):
    # Load the current frame and convert to grayscale
    curr_image_path = os.path.join(sequence_folder, "in{:06d}.jpg".format(i))
    curr_image = cv2.imread(curr_image_path)
    curr_gray = cv2.cvtColor(curr_image, cv2.COLOR_BGR2GRAY)

    # Background subtraction using mean approximation
    if BUF_mean is None:
        BUF_mean = np.float32(curr_gray)
    else:
        BUF_mean = alpha * curr_gray + (1 - alpha) * BUF_mean

    # Convert background model to uint8
    bg_mean = np.uint8(BUF_mean)

    # Perform binarization for mean approximation
    _, binarized_image_mean = cv2.threshold(cv2.absdiff(curr_gray, bg_mean), 10, 255, cv2.THRESH_BINARY)

    # Background subtraction using median approximation
    if BGN_median is None:
        BGN_median = curr_gray
    else:
        BGN_median = np.where(BGN_median < curr_gray, BGN_median + 1, np.where(BGN_median > curr_gray, BGN_median - 1, BGN_median))

    # Convert background model to uint8 for median approximation
    bg_median = np.uint8(BGN_median)

    # Perform binarization for median approximation
    _, binarized_image_median = cv2.threshold(cv2.absdiff(curr_gray, bg_median), 10, 255, cv2.THRESH_BINARY)

    # Load the ground truth mask
    groundtruth_path = os.path.join(r"c:\Users\anton\Documents\Erasmus\AVS\Lab2\pedestrian", "groundtruth", "gt{:06d}.png".format(i))
    groundtruth_mask = cv2.imread(groundtruth_path, cv2.IMREAD_GRAYSCALE)

    # Perform pixel-wise comparison between binarized images and ground truth mask
    if i >= start_frame and i <= end_frame:
        # For mean approximation
        TP_M_mean = np.logical_and((binarized_image_mean == 255), (groundtruth_mask == 255))
        FP_M_mean = np.logical_and((binarized_image_mean == 255), (groundtruth_mask == 0))
        FN_M_mean = np.logical_and((binarized_image_mean == 0), (groundtruth_mask == 255))

        TP_mean += np.sum(TP_M_mean)
        FP_mean += np.sum(FP_M_mean)
        FN_mean += np.sum(FN_M_mean)

        # For median approximation
        TP_M_median = np.logical_and((binarized_image_median == 255), (groundtruth_mask == 255))
        FP_M_median = np.logical_and((binarized_image_median == 255), (groundtruth_mask == 0))
        FN_M_median = np.logical_and((binarized_image_median == 0), (groundtruth_mask == 255))

        TP_median += np.sum(TP_M_median)
        FP_median += np.sum(FP_M_median)
        FN_median += np.sum(FN_M_median)

# Compute evaluation metrics for mean approximation
precision_mean = TP_mean / (TP_mean + FP_mean) if (TP_mean + FP_mean) > 0 else 0
recall_mean = TP_mean / (TP_mean + FN_mean) if (TP_mean + FN_mean) > 0 else 0
f1_score_mean = 2 * precision_mean * recall_mean / (precision_mean + recall_mean) if (precision_mean + recall_mean) > 0 else 0

# Compute evaluation metrics for median approximation
precision_median = TP_median / (TP_median + FP_median) if (TP_median + FP_median) > 0 else 0
recall_median = TP_median / (TP_median + FN_median) if (TP_median + FN_median) > 0 else 0
f1_score_median = 2 * precision_median * recall_median / (precision_median + recall_median) if (precision_median + recall_median) > 0 else 0

# Print evaluation metrics for mean approximation
print("Buffer Mean Method:")
print("Precision (P):", precision_mean)
print("Recall (R):", recall_mean)

# Print evaluation metrics for the median method
print("Buffer Median Method:")
print("Precision (P):", precision_median)
print("Recall (R):", recall_median)
print("F1 Score:", f1_score_median)