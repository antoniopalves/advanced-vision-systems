import cv2
import numpy as np

def optical_flow_block_method(frame1, frame2, block_size=5, search_size=3):
    # Convert frames to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Calculate the shape of the frames
    height, width = gray1.shape

    # Initialize empty arrays to store optical flow vectors
    flow_x = np.zeros_like(gray1, dtype=np.float32)
    flow_y = np.zeros_like(gray1, dtype=np.float32)

    # Loop through each pixel in the frame
    for y in range(block_size+search_size, height - (block_size+search_size)):
        for x in range(block_size+search_size, width - (block_size+search_size)):
            # Define the search window
            patch1 = gray1[y - block_size:y + block_size + 1, x - block_size:x + block_size + 1].astype(np.float32)
            #print("Patch 1 shape:", patch1.shape)
            min_sad = float('inf')
            best_dx = 0
            best_dy = 0

            # Search for the best matching block in the second frame
            for dy in range(-search_size, search_size + 1):
                for dx in range(-search_size, search_size + 1):
                    # Calculate the Sum of Absolute Differences
                    patch2 = gray2[y - block_size+dy:y + block_size + 1+dy, x - block_size+dx:x + block_size + 1+dx].astype(np.float32)
                    #print("Patch 2 shape:", patch2.shape)
                    sad = np.sum(np.square(patch1 - patch2))

                    if sad < min_sad:
                        min_sad = sad
                        best_dx = dx
                        best_dy = dy
                        
            # Store the optical flow vectors
            flow_x[y, x] = np.float32(best_dx)
            flow_y[y, x] = np.float32(best_dy)

    return flow_x, flow_y

def vis_flow(u, v, YX, name):
    # Convert flow vectors to polar coordinates
    magnitude, angle = cv2.cartToPolar(u, v)
    # Resize angle array to match frame dimensions
    angle_resized = cv2.resize(angle, (YX[1], YX[0]))
    # Resize magnitude array to match frame dimensions
    magnitude_resized = cv2.resize(magnitude, (YX[1], YX[0]))
    # Convert to HSV color space
    hsv = np.zeros((YX[0], YX[1], 3), dtype=np.uint8)
    hsv[..., 0] = angle_resized * 90 / np.pi
    hsv[..., 1] = 255
    # Normalize the resized magnitude array
    magnitude_normalized = cv2.normalize(magnitude_resized, None, 0, 255, cv2.NORM_MINMAX)
    hsv[..., 2] = magnitude_normalized
    # Convert HSV to BGR
    flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    # Display
    cv2.imshow(name, flow_rgb)


def of(I_org, I, J, W2=3, dY=3, dX=3):
    # Calculate optical flow using block method
    flow_x, flow_y = optical_flow_block_method(I, J, block_size=7, search_size=3)
    return flow_x, flow_y

def pyramid(im, max_scale):
    images = [im]  # List to store pyramid images
    for k in range(1, max_scale):
        downscaled_image = cv2.resize(images[k - 1], None, fx=0.5, fy=0.5)
        print(f"Downscaled image {k} dimensions: {downscaled_image.shape}")
        images.append(downscaled_image)
    return images

# Load the images
frame1 = cv2.imread(r'c:\Users\anton\Documents\Erasmus\AVS\Lab4\I.jpg')
frame2 = cv2.imread(r'c:\Users\anton\Documents\Erasmus\AVS\Lab4\J.jpg')

# Calculate pyramid for each image
pyramid_frame1 = pyramid(frame1, 3)
pyramid_frame2 = pyramid(frame2, 3)

# Calculate optical flow for each scale
for scale in range(len(pyramid_frame1)):
    if scale == 0:
        flow_x_scale, flow_y_scale = of(frame1, pyramid_frame1[scale], pyramid_frame2[scale])
    else:
        flow_x_scale, flow_y_scale = of(pyramid_frame1[scale-1], pyramid_frame1[scale], pyramid_frame2[scale])
    vis_flow(flow_x_scale, flow_y_scale, frame1.shape[:2], f'Optical Flow Scale {scale}')
    
cv2.waitKey(0)
cv2.destroyAllWindows()
