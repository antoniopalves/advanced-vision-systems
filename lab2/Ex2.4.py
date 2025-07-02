import cv2
import os
import numpy as np

# Set the path to the folder containing the image sequence
sequence_folder = r"c:\Users\anton\Documents\Erasmus\AVS\Lab2\pedestrian\input" #can be changed toto highway or office - just switch pedestrian with keyword

# Specify the start and end frame indices
start_frame = 300
end_frame = 1100

# Specify the step for frame analysis
step = 1  # Adjust this value as needed

# Threshold value for binarization
threshold_value = 9  # Adjust this value as needed

# Initialize counters for evaluation metrics
TP = 0  # True Positive
FP = 0  # False Positive
FN = 0  # False Negative

for i in range(start_frame + 1, end_frame, step):
    # Load the current frame and convert to grayscale
    curr_image_path = os.path.join(sequence_folder, "in{:06d}.jpg".format(i))
    curr_image = cv2.imread(curr_image_path)
    curr_gray = cv2.cvtColor(curr_image, cv2.COLOR_BGR2GRAY)
    
    if curr_gray is None:
        print("Error loading image:", curr_image_path)
        continue
    
    # Load the previous frame and convert to grayscale
    prev_image_path = os.path.join(sequence_folder, "in{:06d}.jpg".format(i - 1))
    prev_image = cv2.imread(prev_image_path)
    prev_gray = cv2.cvtColor(prev_image, cv2.COLOR_BGR2GRAY)
    
    if prev_gray is None:
        print("Error loading image:", prev_image_path)
        continue
    
    # Subtract the current grayscale image from the previous one
    diff_image = cv2.absdiff(curr_gray, prev_gray)
    
    # Median filtering to reduce noise
    diff_image = cv2.medianBlur(diff_image, 5)  # Adjust kernel size as needed
    
    # Binarization
    _, binarized_image = cv2.threshold(diff_image, threshold_value, 255, cv2.THRESH_BINARY)
    
    # Perform erosion followed by dilation to refine the silhouette
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # Define the kernel
    binarized_image = cv2.erode(binarized_image, kernel)
    binarized_image = cv2.dilate(binarized_image, kernel)
    
    # Load the ground truth mask
    groundtruth_path = os.path.join(r"c:\Users\anton\Documents\Erasmus\AVS\Lab2\pedestrian", "groundtruth", "gt{:06d}.png".format(i)) #can be changed toto highway or office - just switch pedestrian with keyword
    groundtruth_mask = cv2.imread(groundtruth_path, cv2.IMREAD_GRAYSCALE)
    
    if groundtruth_mask is None:
        print("Error loading ground truth mask:", groundtruth_path)
        continue
    
    # Perform pixel-wise comparison between binarized image and ground truth mask
    if i >= start_frame and i <= end_frame:  # Check if frame is within specified range
        TP_M = np.logical_and((binarized_image == 255), (groundtruth_mask == 255))
        FP_M = np.logical_and((binarized_image == 255), (groundtruth_mask == 0))
        FN_M = np.logical_and((binarized_image == 0), (groundtruth_mask == 255))
        
        TP += np.sum(TP_M)
        FP += np.sum(FP_M)
        FN += np.sum(FN_M)

# Compute evaluation metrics
P = TP / (TP + FP) if (TP + FP) > 0 else 0
R = TP / (TP + FN) if (TP + FN) > 0 else 0
F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0

# Print evaluation metrics
print("Precision (P):", P)
print("Recall (R):", R)
print("F1 Score:", F1)
