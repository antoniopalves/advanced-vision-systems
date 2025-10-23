import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg' for PyQt5
import matplotlib.pyplot as plt

from scipy.ndimage import sobel, gaussian_filter, maximum_filter
import pm

def harris_response(image_gray, sobel_size, gauss_size, k):
    Ix = sobel(image_gray, axis=1)
    Iy = sobel(image_gray, axis=0)
    Ixx = gaussian_filter(Ix**2, gauss_size)
    Ixy = gaussian_filter(Ix*Iy, gauss_size)
    Iyy = gaussian_filter(Iy**2, gauss_size)
    
    det_M = Ixx * Iyy - Ixy**2
    trace_M = Ixx + Iyy
    H = det_M - k * (trace_M**2)
    
    return H

def find_max(image, size, threshold):
    data_max = maximum_filter(image, size)
    maxima = (image == data_max)
    diff = image > threshold
    maxima[diff == 0] = 0
    return np.nonzero(maxima)

def plot_points(image, points):
    plt.imshow(image, cmap='gray')
    plt.plot(points[1], points[0], '*', color='r')
    plt.show()

def feature_descriptions(image, feature_points, patch_size):
    X, Y = image.shape
    # Filter out points whose neighborhoods don't fit into the image
    feature_points = list(filter(lambda pt: pt[0] >= patch_size and pt[0] < Y - patch_size and pt[1] >= patch_size and pt[1] < X - patch_size, feature_points))
    # Create descriptions of feature points
    descriptions = []
    for point in feature_points:
        x, y = point
        patch = image[y - patch_size:y + patch_size + 1, x - patch_size:x + patch_size + 1]
        descriptions.append((patch.flatten(), point))
    return descriptions

def find_similar(descriptions1, descriptions2, n):
    matches = []
    for desc1 in descriptions1:
        min_dist = float('inf')
        min_desc = None
        for desc2 in descriptions2:
            dist = np.linalg.norm(desc1[0] - desc2[0])  # Euclidean distance
            if dist < min_dist:
                min_dist = dist
                min_desc = desc2
        matches.append((desc1[1], min_desc[1], min_dist))
    matches.sort(key=lambda x: x[2])  # Sort by distance
    return matches[:n]

def feature_descriptions_affine(image, feature_points, patch_size):
    X, Y = image.shape
    descriptions = []
    for i in range(len(feature_points[0])):
        x, y = feature_points[1][i], feature_points[0][i]
        if x >= patch_size and x < Y - patch_size and y >= patch_size and y < X - patch_size:
            patch = image[y - patch_size:y + patch_size + 1, x - patch_size:x + patch_size + 1]
            mean_intensity = np.mean(patch)
            std_intensity = np.std(patch)
            patch = (patch - mean_intensity) / std_intensity
            descriptions.append((patch.flatten(), (x, y)))
    return descriptions

# Load images
fontanna1 = plt.imread(r'c:\Users\anton\Documents\Erasmus\AVS\Lab6\fontanna1.jpg')
fontanna2 = plt.imread(r'c:\Users\anton\Documents\Erasmus\AVS\Lab6\fontanna2.jpg')
budynek1 = plt.imread(r'c:\Users\anton\Documents\Erasmus\AVS\Lab6\budynek1.jpg')
budynek2 = plt.imread(r'c:\Users\anton\Documents\Erasmus\AVS\Lab6\budynek2.jpg')

# Convert images to grayscale
fontanna1_gray = np.mean(fontanna1, axis=2)
fontanna2_gray = np.mean(fontanna2, axis=2)
budynek1_gray = np.mean(budynek1, axis=2)
budynek2_gray = np.mean(budynek2, axis=2)

# Parameters
sobel_size = 3
gauss_size = 3
k = 0.05
threshold = 0.01
mask_size = 7
patch_size = 15
n_matches = 20

# Apply Harris corner detection
H_fontanna1 = harris_response(fontanna1_gray, sobel_size, gauss_size, k)
H_fontanna2 = harris_response(fontanna2_gray, sobel_size, gauss_size, k)
H_budynek1 = harris_response(budynek1_gray, sobel_size, gauss_size, k)
H_budynek2 = harris_response(budynek2_gray, sobel_size, gauss_size, k)

# Find local maxima
maxima_fontanna1 = find_max(H_fontanna1, mask_size, threshold)
maxima_fontanna2 = find_max(H_fontanna2, mask_size, threshold)
maxima_budynek1 = find_max(H_budynek1, mask_size, threshold)
maxima_budynek2 = find_max(H_budynek2, mask_size, threshold)

# Create descriptions of feature points
descriptions_fontanna1 = feature_descriptions_affine(fontanna1_gray, maxima_fontanna1, patch_size)
descriptions_fontanna2 = feature_descriptions_affine(fontanna2_gray, maxima_fontanna2, patch_size)
descriptions_budynek1 = feature_descriptions_affine(budynek1_gray, maxima_budynek1, patch_size)
descriptions_budynek2 = feature_descriptions_affine(budynek2_gray, maxima_budynek2, patch_size)

# Find best matches
matches_fontanna = find_similar(descriptions_fontanna1, descriptions_fontanna2, n_matches)
matches_budynek = find_similar(descriptions_budynek1, descriptions_budynek2, n_matches)

# After finding matches, add print statements to check if matches are found
print("Number of matches in fontanna:", len(matches_fontanna))
print("Number of matches in budynek:", len(matches_budynek))

# Add print statements to check the content of matches
print("Matches in fontanna:", matches_fontanna)
print("Matches in budynek:", matches_budynek)

# Add print statements before and after calling plotting functions
print("Plotting matches for fontanna...")
# Plot matches
pm.plot_matches(fontanna1_gray, fontanna2_gray, matches_fontanna)
print("Plotting matches for budynek...")
pm.plot_matches(budynek1_gray, budynek2_gray, matches_budynek)
print("Plotting complete.")

plt.show()