import cv2
import numpy as np
from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score

# Function to compute HOG descriptor using OpenCV
def compute_hog_descriptor(image):
    win_size = (64, 128)
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size = (8, 8)
    nbins = 9
    
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    hog_descriptor = hog.compute(image)
    
    return hog_descriptor.flatten()

# Step 1: Prepare dataset and compute HOG descriptors
HOG_data = []
labels = []

# Load positive samples (pedestrians)
for i in range(1, 924):  # Adjust the range according to your dataset
    image_path = rf'c:\Users\anton\Documents\Erasmus\AVS\Lab11\pos\per{i:05}.ppm'
    image = cv2.imread(image_path)
    
    if image is not None:
        hog_descriptor = compute_hog_descriptor(image)
        HOG_data.append(hog_descriptor)
        labels.append(1)  # Class label 1 for pedestrians
    else:
        print(f"Failed to load image: {image_path}")

# Load negative samples (non-pedestrians)
for i in range(0, 924):  # Adjust the range according to your dataset
    image_path = rf'c:\Users\anton\Documents\Erasmus\AVS\Lab11\neg\neg{i:05}.png'
    image = cv2.imread(image_path)
    
    if image is not None:
        hog_descriptor = compute_hog_descriptor(image)
        HOG_data.append(hog_descriptor)
        labels.append(0)  # Class label 0 for non-pedestrians
    else:
        print(f"Failed to load image: {image_path}")

HOG_data = np.array(HOG_data, dtype=np.float32)
labels = np.array(labels, dtype=np.int32)

# Step 2: Initialize SVM classifier
clf = svm.SVC(kernel='linear', C=1.0)

# Step 3: Train the SVM classifier
clf.fit(HOG_data, labels)

# Step 4: Predict on the training set for evaluation
predictions_train = clf.predict(HOG_data)

# Step 5: Evaluate performance on training set
accuracy_train = accuracy_score(labels, predictions_train)
conf_matrix_train = confusion_matrix(labels, predictions_train)

print("Training Set Performance:")
print(f"Accuracy: {accuracy_train * 100:.2f}%")
print("Confusion Matrix:")
print(conf_matrix_train)

# Optional: Save the trained model for later use
import joblib
model_path = r'c:\Users\anton\Documents\Erasmus\AVS\Lab11\svm_pedestrian_detector.pkl'
joblib.dump(clf, model_path)
print(f"Model saved to {model_path}")

# Optional: Load the trained model
# clf = joblib.load(model_path)

# Step 6: Implement pedestrian detection on a test image
def detect_pedestrians(image, clf, win_stride=(8, 8), scale=1.05):
    detections = []
    (h, w) = image.shape[:2]
    
    for scale_factor in np.arange(1, 2, 0.1):  # Adjust scale range for better detection
        resized_image = cv2.resize(image, (int(w / scale_factor), int(h / scale_factor)))
        for y in range(0, resized_image.shape[0] - 128, win_stride[1]):
            for x in range(0, resized_image.shape[1] - 64, win_stride[0]):
                window = resized_image[y:y + 128, x:x + 64]
                hog_descriptor = compute_hog_descriptor(window)
                hog_descriptor = hog_descriptor.reshape(1, -1)
                prediction = clf.predict(hog_descriptor)
                if prediction == 1:
                    detections.append((int(x * scale_factor), int(y * scale_factor),
                                       int((x + 64) * scale_factor), int((y + 128) * scale_factor)))
    
    return detections

# Load a test image and detect pedestrians
test_image_path = r'c:\Users\anton\Documents\Erasmus\AVS\Lab11\testImage1.png' 
test_image = cv2.imread(test_image_path)
if test_image is not None:
    detections = detect_pedestrians(test_image, clf)
    for (x1, y1, x2, y2) in detections:
        cv2.rectangle(test_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Display the detection results
    cv2.imshow('Pedestrian Detection', test_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print(f"Failed to load test image: {test_image_path}")
