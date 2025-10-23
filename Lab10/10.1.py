import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load the video sequence
cap = cv2.VideoCapture(r'c:\Users\anton\Documents\Erasmus\AVS\Lab10\vid1_IR.avi')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale
    G = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Step 3: Binarisation using a fixed threshold
    _, binary = cv2.threshold(G, 128, 255, cv2.THRESH_BINARY)  # Adjust threshold value as needed

    # Step 4: Filtering
    # Applying median filter
    binary = cv2.medianBlur(binary, 5)

    # Applying morphological operations
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Step 5: Labelling
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    # Step 6: Analysis of labelling results
    for i in range(1, num_labels):  # Start from 1 to skip the background
        x, y, w, h, area = stats[i]
        if area > 500:  # Filter objects based on size (adjust threshold as needed)
            aspect_ratio = h / w
            if aspect_ratio > 1.2:  # Filter objects based on shape (vertical silhouettes)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the result
    cv2.imshow('IR', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
