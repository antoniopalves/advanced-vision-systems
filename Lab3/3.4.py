import cv2
import os
import numpy as np

# Create a BackgroundSubtractorMOG2 object
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)

# Specify the path to the folder containing the image sequence
sequence_folder = r"c:\Users\anton\Documents\Erasmus\AVS\Lab2\pedestrian\input" 

# Specify the start and end frame indices
start_frame = 300
end_frame = 1000

# Initialize variables for evaluation metrics
TP = 0
FP = 0
FN = 0

# Main loop for processing each frame
for i in range(start_frame, end_frame + 1):
    # Read the current frame
    frame_path = os.path.join(sequence_folder, "in{:06d}.jpg".format(i))
    frame = cv2.imread(frame_path)

    # Apply background subtraction
    fg_mask = bg_subtractor.apply(frame)

    cv2.imshow('mask',fg_mask)
    cv2.waitKey(1)

    # Load the ground truth mask for comparison
    ground_truth_mask_path = os.path.join(r"c:\Users\anton\Documents\Erasmus\AVS\Lab2\pedestrian", "groundtruth", "gt{:06d}.png".format(i))
    ground_truth_mask = cv2.imread(ground_truth_mask_path, cv2.IMREAD_GRAYSCALE)

    # Perform pixel-wise comparison between foreground mask and ground truth mask
    TP += np.sum(np.logical_and(fg_mask == 255, ground_truth_mask == 255))
    FP += np.sum(np.logical_and(fg_mask == 255, ground_truth_mask == 0))
    FN += np.sum(np.logical_and(fg_mask == 0, ground_truth_mask == 255))

# Calculate precision, recall, and F1 score
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

# Print the evaluation metrics
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)
