import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load images
image_left = cv2.imread(r'c:\Users\anton\Documents\Erasmus\AVS\Lab6\left_panorama.jpg', cv2.IMREAD_GRAYSCALE)
image_right = cv2.imread(r'c:\Users\anton\Documents\Erasmus\AVS\Lab6\right_panorama.jpg', cv2.IMREAD_GRAYSCALE)

# Find feature points in the images
orb = cv2.ORB_create()
keypoints_left, descriptors_left = orb.detectAndCompute(image_left, None)
keypoints_right, descriptors_right = orb.detectAndCompute(image_right, None)

# Display feature points
image_left_with_keypoints = cv2.drawKeypoints(image_left, keypoints_left, None, color=(255, 0, 0), flags=0)
image_right_with_keypoints = cv2.drawKeypoints(image_right, keypoints_right, None, color=(255, 0, 0), flags=0)

# Convert images to RGB format for matplotlib
image_left_rgb = cv2.cvtColor(image_left_with_keypoints, cv2.COLOR_BGR2RGB)
image_right_rgb = cv2.cvtColor(image_right_with_keypoints, cv2.COLOR_BGR2RGB)

# Display images with feature points
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(image_left_rgb)
ax[0].set_title('Left Panorama with Feature Points')
ax[0].axis('off')
ax[1].imshow(image_right_rgb)
ax[1].set_title('Right Panorama with Feature Points')
ax[1].axis('off')
plt.show()

# Match feature points
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(descriptors_left, descriptors_right)

# Sort matches by distance
matches = sorted(matches, key=lambda x: x.distance)

# Select the best matches
best_matches = [m for m in matches if m.distance < 0.5 * m.distance]

# Draw matches
image_matches = cv2.drawMatches(image_left, keypoints_left, image_right, keypoints_right, best_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Convert image to RGB format for matplotlib
image_matches_rgb = cv2.cvtColor(image_matches, cv2.COLOR_BGR2RGB)

# Display matches
plt.figure(figsize=(10, 6))
plt.imshow(image_matches_rgb)
plt.title('Best Matches between Left and Right Panorama')
plt.axis('off')
plt.show()
