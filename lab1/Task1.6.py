import cv2
import matplotlib.pyplot as plt

# Read the grayscale image
I = cv2.imread(r"C:\Users\anton\Documents\Erasmus\AVS\mandrill.jpg", cv2.IMREAD_GRAYSCALE)

# Perform classical histogram equalization
IGE = cv2.equalizeHist(I)

# Create a CLAHE object
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# Perform CLAHE histogram equalization
I_CLAHE = clahe.apply(I)

# Display the original, classical histogram equalization, and CLAHE equalization results
plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1), plt.imshow(I, cmap='gray'), plt.title("Original")
plt.subplot(2, 2, 2), plt.hist(I.ravel(), 256, [0, 256]), plt.title("Original Histogram")
plt.subplot(2, 2, 3), plt.imshow(IGE, cmap='gray'), plt.title("Classical Equalization")
plt.subplot(2, 2, 4), plt.hist(IGE.ravel(), 256, [0, 256]), plt.title("Classical Equalization Histogram")

plt.show()

# Display the CLAHE equalization results
plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1), plt.imshow(I, cmap='gray'), plt.title("Original")
plt.subplot(2, 2, 2), plt.hist(I.ravel(), 256, [0, 256]), plt.title("Original Histogram")
plt.subplot(2, 2, 3), plt.imshow(I_CLAHE, cmap='gray'), plt.title("CLAHE Equalization")
plt.subplot(2, 2, 4), plt.hist(I_CLAHE.ravel(), 256, [0, 256]), plt.title("CLAHE Equalization Histogram")

plt.show()
