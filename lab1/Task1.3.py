import cv2
import matplotlib.pyplot as plt

# Read the mandrill image
I = cv2.imread(r"C:\Users\anton\Documents\Erasmus\AVS\mandrill.jpg")

# Convert the image to HSV space
IHSV = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)

# Extract individual components
IH = IHSV[:, :, 0]
IS = IHSV[:, :, 1]
IV = IHSV[:, :, 2]

# Display the original image and individual HSV components
plt.figure(figsize=(10, 4))

# Original image
plt.subplot(1, 4, 1)
plt.imshow(cv2.cvtColor(I, cv2.COLOR_BGR2RGB))
plt.title("Original")
plt.axis("off")

# Hue component
plt.subplot(1, 4, 2)
plt.imshow(IH, cmap='hsv')
plt.title("Hue")
plt.axis("off")

# Saturation component
plt.subplot(1, 4, 3)
plt.imshow(IS, cmap='gray')
plt.title("Saturation")
plt.axis("off")

# Value component
plt.subplot(1, 4, 4)
plt.imshow(IV, cmap='gray')
plt.title("Value")
plt.axis("off")

plt.tight_layout()
plt.show()
