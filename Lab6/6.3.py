import numpy as np
import matplotlib.pyplot as plt
import cv2
import random

def harris_measure(image, ksize, k):
    """Compute Harris corner measure."""
    dx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize)
    dy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize)
    dx2 = dx ** 2
    dy2 = dy ** 2
    dxy = dx * dy
    dx2_blur = cv2.GaussianBlur(dx2, (5, 5), 0)
    dy2_blur = cv2.GaussianBlur(dy2, (5, 5), 0)
    dxy_blur = cv2.GaussianBlur(dxy, (5, 5), 0)
    det_M = dx2_blur * dy2_blur - dxy_blur ** 2
    trace_M = dx2_blur + dy2_blur
    return det_M - k * trace_M ** 2

def non_max_suppression(keypoints, window_size):
    """Perform non-maximum suppression."""
    keypoints_sorted = sorted(keypoints, key=lambda x: x.response, reverse=True)
    selected_keypoints = []
    for kp in keypoints_sorted:
        if not any(abs(kp.pt[0] - skp.pt[0]) < window_size and abs(kp.pt[1] - skp.pt[1]) < window_size for skp in selected_keypoints):
            selected_keypoints.append(kp)
    return selected_keypoints

def intensity_centroid_orientation(image, center, patch_size):
    """Calculate intensity centroid and orientation."""
    patch = image[int(center[1]) - patch_size:int(center[1]) + patch_size + 1,
                  int(center[0]) - patch_size:int(center[0]) + patch_size + 1]
    cx = np.sum(patch * np.arange(patch.shape[1])) / np.sum(patch)
    cy = np.sum(patch * np.arange(patch.shape[0])) / np.sum(patch)
    orientation = np.arctan2(cy - patch_size, cx - patch_size)
    return (cx, cy), orientation

def generate_brief_descriptor(image, center, patch_size, orientation):
    """Generate BRIEF descriptor."""
    patch = image[int(center[1]) - patch_size:int(center[1]) + patch_size + 1,
                  int(center[0]) - patch_size:int(center[0]) + patch_size + 1]
    rotated_patch = cv2.warpAffine(patch, cv2.getRotationMatrix2D((patch_size, patch_size), np.degrees(orientation), 1), (2 * patch_size + 1, 2 * patch_size + 1))
    descriptors = []
    pairs = [(random.randint(0, 2 * patch_size), random.randint(0, 2 * patch_size)) for _ in range(256)]
    for p1, p2 in pairs:
        descriptors.append(1 if rotated_patch[p1 // 2, p1 % 2] < rotated_patch[p2 // 2, p2 % 2] else 0)
    return descriptors

def hamming_distance(descriptor1, descriptor2):
    """Calculate Hamming distance."""
    return sum(a != b for a, b in zip(descriptor1, descriptor2))

def display_keypoints(image, keypoints, title):
    """Display image with keypoints."""
    plt.imshow(cv2.drawKeypoints(image, keypoints, None, flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS))
    plt.title(title)
    plt.show()

# Load images
fontanna1 = cv2.imread(r'c:\Users\anton\Documents\Erasmus\AVS\Lab6\fontanna1.jpg', cv2.IMREAD_GRAYSCALE)
fontanna2 = cv2.imread(r'c:\Users\anton\Documents\Erasmus\AVS\Lab6\fontanna2.jpg', cv2.IMREAD_GRAYSCALE)

# Step 1: FAST Keypoint Detection
fast = cv2.FastFeatureDetector_create()

keypoints1 = fast.detect(fontanna1, None)
keypoints2 = fast.detect(fontanna2, None)

# Step 2: Harris Measure Computation
ksize = 3
k = 0.04

keypoints1_harris = [(kp, harris_measure(fontanna1, ksize, k)[int(kp.pt[1]), int(kp.pt[0])]) for kp in keypoints1]
keypoints1_filtered = non_max_suppression([kp[0] for kp in keypoints1_harris], window_size=3)

keypoints2_harris = [(kp, harris_measure(fontanna2, ksize, k)[int(kp.pt[1]), int(kp.pt[0])]) for kp in keypoints2]
keypoints2_filtered = non_max_suppression([kp[0] for kp in keypoints2_harris], window_size=3)

# Step 3: Keypoint Filtering with Non-Maximum Suppression
N = 500

keypoints1_selected = keypoints1_filtered[:N]
keypoints2_selected = keypoints2_filtered[:N]

# Step 4: Remove Keypoints without Full Descriptor Environment
patch_size = 15

keypoints1_selected = [kp for kp in keypoints1_selected if 0 <= kp.pt[0] - patch_size and kp.pt[0] + patch_size < fontanna1.shape[1] and 0 <= kp.pt[1] - patch_size and kp.pt[1] + patch_size < fontanna1.shape[0]]
keypoints2_selected = [kp for kp in keypoints2_selected if 0 <= kp.pt[0] - patch_size and kp.pt[0] + patch_size < fontanna2.shape[1] and 0 <= kp.pt[1] - patch_size and kp.pt[1] + patch_size < fontanna2.shape[0]]

# Step 5: Harris Measure Sorting
ksize = 3
k = 0.04

keypoints1_harris = [(kp, harris_measure(fontanna1, ksize, k)[int(kp.pt[1]), int(kp.pt[0])]) for kp in keypoints1_filtered]
keypoints1_harris.sort(key=lambda x: x[1], reverse=True)
keypoints1_selected = [kp[0] for kp in keypoints1_harris[:N]]

keypoints2_harris = [(kp, harris_measure(fontanna2, ksize, k)[int(kp.pt[1]), int(kp.pt[0])]) for kp in keypoints2_filtered]
keypoints2_harris.sort(key=lambda x: x[1], reverse=True)
keypoints2_selected = [kp[0] for kp in keypoints2_harris[:N]]

# Step 6: Centroid and Orientation Calculation
patch_size = 15
keypoints1_oriented = [(kp, intensity_centroid_orientation(fontanna1, kp.pt, patch_size)) for kp in keypoints1_selected]
keypoints2_oriented = [(kp, intensity_centroid_orientation(fontanna2, kp.pt, patch_size)) for kp in keypoints2_selected]

# Step 7: BRIEF Descriptor Generation
descriptors1 = [generate_brief_descriptor(fontanna1, kp[0].pt, patch_size, kp[1][1]) for kp in keypoints1_oriented]
descriptors2 = [generate_brief_descriptor(fontanna2, kp[0].pt, patch_size, kp[1][1]) for kp in keypoints2_oriented]

# Step 8: Hamming Distance Calculation
matches = []
for i, desc1 in enumerate(descriptors1):
    for j, desc2 in enumerate(descriptors2):
        matches.append((i, j, hamming_distance(desc1, desc2)))

matches.sort(key=lambda x: x[2])

# Step 9: Select N Best Matches
N = 20
best_matches = matches[:N]

# Display results
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
display_keypoints(fontanna1, keypoints1_selected, 'Fontanna 1 Key Points')

plt.subplot(1, 2, 2)
display_keypoints(fontanna2, keypoints2_selected, 'Fontanna 2 Key Points')
