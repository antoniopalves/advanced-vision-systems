import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import sobel

def calculate_gradients(image):
    # Convert image to grayscale if it's RGB
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate gradients using Sobel operators
    gradient_x = sobel(image, axis=1)
    gradient_y = sobel(image, axis=0)
    
    # Calculate magnitude and orientation of gradients
    magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    orientation = np.arctan2(gradient_y, gradient_x) * (180 / np.pi)
    
    return magnitude, orientation

def compute_histograms(magnitude, orientation, cell_size=(8, 8), bins=9):
    height, width = magnitude.shape
    cell_height, cell_width = cell_size
    orientation_bins = np.int32(bins)
    
    # Initialize histograms
    histograms = np.zeros((height // cell_height, width // cell_width, orientation_bins))
    
    for y in range(height // cell_height):
        for x in range(width // cell_width):
            # Determine cells
            cell_magnitude = magnitude[y * cell_height:(y + 1) * cell_height,
                                       x * cell_width:(x + 1) * cell_width]
            cell_orientation = orientation[y * cell_height:(y + 1) * cell_height,
                                           x * cell_width:(x + 1) * cell_width]
            
            # Compute histogram for the cell
            hist, _ = np.histogram(cell_orientation, bins=orientation_bins,
                                   range=(0, 180), weights=cell_magnitude)
            
            histograms[y, x, :] = hist
    
    return histograms

def normalize_blocks(histograms, block_size=(2, 2), epsilon=1e-5):
    block_height, block_width = block_size
    YY, XX, bins = histograms.shape
    features = []
    
    for jj in range(YY - block_height + 1):
        for ii in range(XX - block_width + 1):
            block = histograms[jj:jj + block_height, ii:ii + block_width, :]
            block_vector = block.ravel()
            
            # L2 normalization
            norm = np.sqrt(np.sum(block_vector ** 2) + epsilon ** 2)
            block_vector /= norm
            
            features.extend(block_vector)
    
    return np.array(features)

def hog_descriptor(image):
    # Step 1: Calculate gradients
    magnitude, orientation = calculate_gradients(image)
    
    # Step 2: Compute histograms of gradients
    histograms = compute_histograms(magnitude, orientation)
    
    # Step 3: Visualize histograms
    plot_histograms(histograms)
    
    # Step 4: Normalize histograms in blocks
    features = normalize_blocks(histograms)
    
    return features

def plot_histograms(histograms):
    # Plot histograms
    num_bins = histograms.shape[2]
    fig, axs = plt.subplots(1, num_bins, figsize=(15, 3))
    fig.suptitle('Histograms of Gradients', fontsize=16)
    
    for i in range(num_bins):
        axs[i].imshow(histograms[:, :, i], cmap='gray')
        axs[i].axis('off')
        axs[i].set_title(f'Bin {i}')
    
    plt.tight_layout()
    plt.show()

# Example usage:
if __name__ == "__main__":
    # Load the image using OpenCV
    image_path = r'C:\Users\anton\Documents\Erasmus\AVS\Lab11\testImage1.png'
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Unable to load image from {image_path}")
    else:
        # Compute HOG descriptor
        hog_features = hog_descriptor(image)
        
        print("HOG feature vector shape:", hog_features.shape)
