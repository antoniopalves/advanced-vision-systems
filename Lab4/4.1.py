import cv2
import numpy as np

def optical_flow_block_method(frame1, frame2, block_size=5, search_size=3):
    # Convert frames to grayscale and downscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Calculate the shape of the frames
    height, width = gray1.shape

    # Initialize empty arrays to store optical flow vectors
    flow_x = np.zeros_like(gray1, dtype=np.float32)
    flow_y = np.zeros_like(gray2, dtype=np.float32)

    # Loop through each pixel in the frame
    for y in range(block_size+search_size, height - (block_size+search_size)):
        for x in range(block_size+search_size, width - (block_size+search_size)):
            # Define the search window
            patch1 = gray1[y - block_size:y + block_size + 1, x - block_size:x + block_size + 1].astype(np.float32)

            min_sad = float('inf')
            best_dx = 0
            best_dy = 0

            # Search for the best matching block in the second frame
            for dy in range(-search_size, search_size + 1):
                for dx in range(-search_size, search_size + 1):
                    # Calculate the Sum of Absolute Differences
                    patch2 = gray2[y - block_size+dy:y + block_size + 1+dy, x - block_size+dx:x + block_size + 1+dx].astype(np.float32)
                    sad = np.sum(np.square(patch1 - patch2))

                    if sad < min_sad:
                        min_sad = sad
                        best_dx = dx
                        best_dy = dy
                        
            # Store the optical flow vectors
            flow_x[y, x] = np.float32(best_dx)
            flow_y[y, x] = np.float32(best_dy)

    return flow_x, flow_y

# Load the images
frame1 = cv2.imread(r'c:\Users\anton\Documents\Erasmus\AVS\Lab4\I.jpg')
frame2 = cv2.imread(r'c:\Users\anton\Documents\Erasmus\AVS\Lab4\J.jpg')

# Downscale the images
frame1 = cv2.resize(frame1, None, fx=0.5, fy=0.5)
frame2 = cv2.resize(frame2, None, fx=0.5, fy=0.5)

# Calculate optical flow using the block method
flow_x, flow_y = optical_flow_block_method(frame1, frame2)

magnitude, angle = cv2.cartToPolar(flow_x, flow_y)
matrix = np.zeros(frame1.shape, dtype=np.float32)
matrix[...,0]=angle * 90/np.pi
matrix[...,1]=cv2.normalize(magnitude,None,0,255,cv2.NORM_MINMAX)
matrix[...,2]= 255

final_display = cv2.cvtColor(matrix, cv2.COLOR_HSV2BGR)

# Normalize the flow vectors for display
#flow_x_display = cv2.normalize(flow_x, None, 0, 255, cv2.NORM_MINMAX)
#flow_y_display = cv2.normalize(flow_y, None, 0, 255, cv2.NORM_MINMAX)

# Convert flow vectors to uint8
#flow_x_display = flow_x_display.astype(np.uint8)
#flow_y_display = flow_y_display.astype(np.uint8)

# Display the optical flow vectors
#cv2.imshow('Optical Flow X', flow_x_display)
#cv2.imshow('Optical Flow Y', flow_y_display)
cv2.imshow('final_display',final_display)
cv2.waitKey(0)
cv2.destroyAllWindows()
