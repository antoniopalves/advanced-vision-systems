import cv2

# Read the mandrill image
I = cv2.imread(r"C:\Users\anton\Documents\Erasmus\AVS\mandrill.jpg")

# Retrieve the height and width of the original image
height, width = I.shape[:2]

# Scale factor
scale = 1.75

# Resize the image using the scale factor
Ix2 = cv2.resize(I, (int(scale * width), int(scale * height)))

# Display the resized image
cv2.imshow("Big Mandrill", Ix2)
cv2.waitKey(0)
cv2.destroyAllWindows()
