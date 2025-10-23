import cv2
import os
import numpy as np

# Set the path to the folder containing the image sequence
sequence_folder = r"c:\Users\anton\Documents\Erasmus\AVS\Lab3\pedestrian\input"

# Specify the start and end frame indices
start_frame = 300
end_frame = 1100

# Specify the step for frame analysis
step = 1  # Adjust this value as needed

# Threshold value for binarization
threshold_value = 9  # Adjust this value as needed

# Initialize counters for evaluation metrics
TP_mean = TP_median = FP_mean = FP_median = FN_mean = FN_median = 0

# Load the first frame to get its dimensions
first_frame_path = os.path.join(sequence_folder, "in{:06d}.jpg".format(start_frame))
first_frame = cv2.imread(first_frame_path)
first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

# Initialize buffer for storing grayscale frames
N = 60  # Buffer size
buffer = np.zeros((first_gray.shape[0], first_gray.shape[1], N), dtype=np.uint8)
iN = 0  # Pointer to the current position in the buffer

for i in range(start_frame + 1, end_frame, step):
    # Load the current frame and convert to grayscale
    curr_image_path = os.path.join(sequence_folder, "in{:06d}.jpg".format(i))
    curr_image = cv2.imread(curr_image_path)
    curr_gray = cv2.cvtColor(curr_image, cv2.COLOR_BGR2GRAY)
    
    if curr_gray is None:
        print("Error loading image:", curr_image_path)
        continue
    
    # Store the current grayscale frame in the buffer
    buffer[:, :, iN] = curr_gray
    
    # Increment the buffer pointer and wrap around if necessary
    iN = (iN + 1) % N
    
    # Compute the mean and median of the buffer frames
    buffer_mean = np.mean(buffer, axis=2, dtype=np.uint8)
    buffer_median = np.median(buffer, axis=2).astype(np.uint8)
    
    # Compute the difference between the current frame and the background models (mean and median)
    diff_mean = cv2.absdiff(curr_gray, buffer_mean)
    diff_median = cv2.absdiff(curr_gray, buffer_median)
    
    # Binarize the difference images using the threshold value
    _, bin_mean = cv2.threshold(diff_mean, threshold_value, 255, cv2.THRESH_BINARY)
    _, bin_median = cv2.threshold(diff_median, threshold_value, 255, cv2.THRESH_BINARY)
    
    # Load the ground truth mask
    groundtruth_path = os.path.join(r"c:\Users\anton\Documents\Erasmus\AVS\Lab2\pedestrian", "groundtruth", "gt{:06d}.png".format(i))
    groundtruth_mask = cv2.imread(groundtruth_path, cv2.IMREAD_GRAYSCALE)
    
    if groundtruth_mask is None:
        print("Error loading ground truth mask:", groundtruth_path)
        continue
    
    # Perform pixel-wise comparison between the binarized images and the ground truth mask
    TP_mean += np.sum(np.logical_and(bin_mean == 255, groundtruth_mask == 255))
    TP_median += np.sum(np.logical_and(bin_median == 255, groundtruth_mask == 255))
    FP_mean += np.sum(np.logical_and(bin_mean == 255, groundtruth_mask == 0))
    FP_median += np.sum(np.logical_and(bin_median == 255, groundtruth_mask == 0))
    FN_mean += np.sum(np.logical_and(bin_mean == 0, groundtruth_mask == 255))
    FN_median += np.sum(np.logical_and(bin_median == 0, groundtruth_mask == 255))

# Compute evaluation metrics for the buffer mean method
P_mean = TP_mean / (TP_mean + FP_mean) if TP_mean + FP_mean > 0 else 0
R_mean = TP_mean / (TP_mean + FN_mean) if TP_mean + FN_mean > 0 else 0
F1_mean = 2 * P_mean * R_mean / (P_mean + R_mean) if P_mean + R_mean > 0 else 0

# Compute evaluation metrics for the buffer median method
P_median = TP_median / (TP_median + FP_median) if TP_median + FP_median > 0 else 0
R_median = TP_median / (TP_median + FN_median) if TP_median + FN_median > 0 else 0
F1_median = 2 * P_median * R_median / (P_median + R_median) if P_median + R_median > 0 else 0

# Print evaluation metrics for the buffer mean method
print("Buffer Mean Method:")
print("Precision (P):", P_mean)
print("Recall (R):", R_mean)
print("F1 Score:", F1_mean)

# Print evaluation metrics for the buffer median method
print("Buffer Median Method:")
print("Precision (P):", P_median)
print("Recall (R):", R_median)
print("F1 Score:", F1_median)