import cv2

# Set the path to the folder containing the image sequence
sequence_folder = r"c:\Users\anton\Documents\Erasmus\AVS\Lab2\pedestrian\input"

# Specify the start and end frame indices
start_frame = 300
end_frame = 1100

# Specify the step for frame analysis
step = 1  # Adjust this value as needed

for i in range(start_frame, end_frame, step):
    # Load the image
    image_path = sequence_folder + r"\in%06d.jpg" % i
    I = cv2.imread(image_path)
    
    # Display the image
    cv2.imshow("Frame", I)
    
    # Wait for a short duration to display the image
    cv2.waitKey(10)

# Close all OpenCV windows
cv2.destroyAllWindows()
