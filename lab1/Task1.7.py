import cv2
import matplotlib.pyplot as plt

# Read the grayscale image
I = cv2.imread(r"C:\Users\anton\Documents\Erasmus\AVS\mandrill.jpg", cv2.IMREAD_GRAYSCALE)

# Gaussian filtration
I_Gaussian = cv2.GaussianBlur(I, (5, 5), 0)

# Sobel filtration
I_Sobel_X = cv2.Sobel(I, cv2.CV_64F, 1, 0, ksize=3)
I_Sobel_Y = cv2.Sobel(I, cv2.CV_64F, 0, 1, ksize=3)
I_Sobel = cv2.magnitude(I_Sobel_X, I_Sobel_Y)

# Laplacian filter
I_Laplacian = cv2.Laplacian(I, cv2.CV_64F)

# Median filtration
I_Median = cv2.medianBlur(I, 5)

# Display the results
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1), plt.imshow(I, cmap='gray'), plt.title("Original")
plt.subplot(2, 3, 2), plt.imshow(I_Gaussian, cmap='gray'), plt.title("Gaussian Filtration")
plt.subplot(2, 3, 3), plt.imshow(I_Sobel, cmap='gray'), plt.title("Sobel Filtration")
plt.subplot(2, 3, 4), plt.imshow(I_Laplacian, cmap='gray'), plt.title("Laplacian Filter")
plt.subplot(2, 3, 5), plt.imshow(I_Median, cmap='gray'), plt.title("Median Filtration")

plt.show()
