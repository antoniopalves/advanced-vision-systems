import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage.filters as filters

def harris_response(image_gray, sobel_size, gauss_size, k):
    Ix = filters.sobel(image_gray, axis=1)
    Iy = filters.sobel(image_gray, axis=0)
    
    Ixx = filters.gaussian_filter(Ix**2, gauss_size)
    Ixy = filters.gaussian_filter(Ix*Iy, gauss_size)
    Iyy = filters.gaussian_filter(Iy**2, gauss_size)
    
    det_M = Ixx * Iyy - Ixy**2
    trace_M = Ixx + Iyy
    H = det_M - k * (trace_M**2)
    
    return H

def find_max(image, size, threshold):
    data_max = filters.maximum_filter(image, size)
    maxima = (image == data_max)
    diff = image > threshold
    maxima[diff == 0] = 0
    return np.nonzero(maxima)

def plot_points(image, points):
    plt.imshow(image, cmap='gray')
    plt.plot(points[1], points[0], '*', color='r')
    plt.show()

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

# Plot detected points
plot_points(fontanna1_gray, maxima_fontanna1)
plot_points(fontanna2_gray, maxima_fontanna2)
plot_points(budynek1_gray, maxima_budynek1)
plot_points(budynek2_gray, maxima_budynek2)
