import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the mandrill image
mandrill = cv2.imread(r"C:\Users\anton\Documents\Erasmus\AVS\mandrill.jpg")
mandrill_gray = cv2.cvtColor(mandrill, cv2.COLOR_BGR2GRAY)

# Load the lena image and resize it to match the mandrill image dimensions
lena = cv2.imread(r"C:\Users\anton\Documents\Erasmus\AVS\lena.png")
lena_gray = cv2.cvtColor(lena, cv2.COLOR_BGR2GRAY)
lena_gray_resized = cv2.resize(lena_gray, (mandrill_gray.shape[1], mandrill_gray.shape[0]))

# Perform addition, subtraction, and multiplication of grayscale images
addition_result = mandrill_gray + lena_gray_resized
subtraction_result = mandrill_gray - lena_gray_resized
multiplication_result = mandrill_gray * lena_gray_resized

# Perform linear combination of grayscale images
alpha = 0.7
beta = 0.3
linear_combination_result = cv2.addWeighted(mandrill_gray, alpha, lena_gray_resized, beta, 0)

# Perform the modulus of the difference of grayscale images
difference_module_result_manual = np.abs(mandrill_gray - lena_gray_resized)
difference_module_result_opencv = cv2.absdiff(mandrill_gray, lena_gray_resized)

# Display the results
plt.figure(figsize=(12, 8))

plt.subplot(2, 4, 1), plt.imshow(mandrill_gray, cmap='gray'), plt.title("Mandrill")
plt.subplot(2, 4, 2), plt.imshow(lena_gray_resized, cmap='gray'), plt.title("Lena Resized")
plt.subplot(2, 4, 3), plt.imshow(addition_result, cmap='gray'), plt.title("Addition")
plt.subplot(2, 4, 4), plt.imshow(subtraction_result, cmap='gray'), plt.title("Subtraction")
plt.subplot(2, 4, 5), plt.imshow(multiplication_result, cmap='gray'), plt.title("Multiplication")
plt.subplot(2, 4, 6), plt.imshow(linear_combination_result, cmap='gray'), plt.title("Linear Combination")
plt.subplot(2, 4, 7), plt.imshow(difference_module_result_manual, cmap='gray'), plt.title("Difference (Manual)")
plt.subplot(2, 4, 8), plt.imshow(difference_module_result_opencv, cmap='gray'), plt.title("Difference (OpenCV)")

plt.show()
