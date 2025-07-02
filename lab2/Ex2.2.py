import cv2
import os
import numpy as np

# Set the path to the folder containing the image sequence
sequence_folder = r"c:\Users\anton\Documents\Erasmus\AVS\Lab2\pedestrian\input"

# Specify the start and end frame indices
start_frame = 300
end_frame = 1100

# Specify the step for frame analysis
step = 1  # Adjust this value as needed

# Define the kernel for morphological operations
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # Adjust kernel size as needed

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
    _, binarized_image = cv2.threshold(diff_image, 10, 255, cv2.THRESH_BINARY)
    
    # Perform erosion followed by dilation to refine the silhouette
    binarized_image = cv2.erode(binarized_image, kernel)
    binarized_image = cv2.dilate(binarized_image, kernel)
    
    # Display the binarized image
    cv2.imshow("Binarized", binarized_image)
    
    # Wait for a short duration to display the image
    key = cv2.waitKey(100)  # Adjust duration as needed (milliseconds)
    if key == ord('q'):  # Press 'q' to exit the loop
        break

# Close all OpenCV windows
cv2.destroyAllWindows()
