import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter

# Load the pattern image
pattern_img = cv2.imread(r'c:\Users\anton\Documents\Erasmus\AVS\Lab9\trybik.jpg')

# Convert to grayscale
gray_pattern = cv2.cvtColor(pattern_img, cv2.COLOR_BGR2GRAY)

# Binarize the image
_, bin_img = cv2.threshold(gray_pattern, 128, 255, cv2.THRESH_BINARY)

# Negate the image
bin_img = cv2.bitwise_not(bin_img)

# Find contours
contours, hierarchy = cv2.findContours(bin_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

# Select the longest contour
contour = max(contours, key=len)

# Calculate gradients
sobelx = cv2.Sobel(gray_pattern, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(gray_pattern, cv2.CV_64F, 0, 1, ksize=5)

# Gradient magnitude and orientation
gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
gradient_orientation = np.arctan2(sobely, sobelx) * 180 / np.pi

# Normalize gradient magnitude
gradient_magnitude = gradient_magnitude / np.amax(gradient_magnitude)

# Calculate moments
moments = cv2.moments(bin_img, 1)
center_of_gravity = (moments['m10'] / moments['m00'], moments['m01'] / moments['m00'])

# Initialize the R-table
Rtable = [[] for _ in range(360)]

# Fill the R-table
for point in contour:
    y, x = point[0]
    angle = int(gradient_orientation[y, x]) % 360
    r = np.sqrt((x - center_of_gravity[0])**2 + (y - center_of_gravity[1])**2)
    phi = np.arctan2(y - center_of_gravity[1], x - center_of_gravity[0])
    Rtable[angle].append((r, phi))

# Load the target image
target_img = cv2.imread(r'c:\Users\anton\Documents\Erasmus\AVS\Lab9\trybiki2.jpg')

# Convert to grayscale
gray_target = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)

# Calculate gradients in the target image
sobelx_t = cv2.Sobel(gray_target, cv2.CV_64F, 1, 0, ksize=5)
sobely_t = cv2.Sobel(gray_target, cv2.CV_64F, 0, 1, ksize=5)
gradient_magnitude_t = np.sqrt(sobelx_t**2 + sobely_t**2)
gradient_orientation_t = np.arctan2(sobely_t, sobelx_t) * 180 / np.pi
gradient_magnitude_t = gradient_magnitude_t / np.amax(gradient_magnitude_t)

# Initialize Hough space
hough_space = np.zeros_like(gray_target)

# Define the number of patterns to detect
num_patterns_to_detect = 5

# Iterate to detect multiple patterns
for _ in range(num_patterns_to_detect):
    # Fill the Hough space
    for y in range(gradient_magnitude_t.shape[0]):
        for x in range(gradient_magnitude_t.shape[1]):
            if gradient_magnitude_t[y, x] > 0.5:
                angle = int(gradient_orientation_t[y, x]) % 360
                for r, phi in Rtable[angle]:
                    x1 = int(x - r * np.cos(phi))
                    y1 = int(y - r * np.sin(phi))
                    if 0 <= x1 < hough_space.shape[1] and 0 <= y1 < hough_space.shape[0]:
                        hough_space[y1, x1] += 1

    # Use a maximum filter to find local maxima in Hough space
    footprint = np.ones((40, 40))  # Define the size of the neighborhood
    hough_space_max = maximum_filter(hough_space, footprint=footprint)
    peaks = (hough_space == hough_space_max) & (hough_space > np.max(hough_space) * 0.35)

    # Find coordinates of the peaks
    peak_coords = np.column_stack(np.where(peaks))

    # Mark the found patterns and draw the contour in red
    for max_y, max_x in peak_coords:
        # Draw a small circle at the detected center
        cv2.circle(target_img, (max_x, max_y), 2, (0, 0, 255), -1)

        # Draw the contour of the detected pattern in red
        for point in contour:
            y, x = point[0]
            r = np.sqrt((x - center_of_gravity[0])**2 + (y - center_of_gravity[1])**2)
            phi = np.arctan2(y - center_of_gravity[1], x - center_of_gravity[0])
            x1 = int(max_x + r * np.cos(phi))
            y1 = int(max_y + r * np.sin(phi))
            if 0 <= x1 < target_img.shape[1] and 0 <= y1 < target_img.shape[0]:
                target_img[y1, x1] = (0, 0, 255)

# Display the Hough space
plt.imshow((hough_space*255).astype(np.uint), cmap='gray')
plt.title('Hough Space')
plt.show()

# Display the result
cv2.imshow('Detected Patterns', target_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
