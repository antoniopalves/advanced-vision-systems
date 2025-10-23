import cv2
import numpy as np

# Termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_FIX_SKEW

# Inner size of chessboard
width = 9
height = 6
square_size = 0.025  # 0.025 meters

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0), ....,(8,6,0)
objp = np.zeros((height * width, 1, 3), np.float32)
objp[:, 0, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)
objp = objp * square_size  # Create real world coordinates. Use your metric.

# Arrays to store object points and image points from all the images
objpoints = []  # 3D point in real world space
imgpointsLeft = []  # 2D points in left image plane
imgpointsRight = []  # 2D points in right image plane

# Load calibration images
path = r"c:\Users\anton\Documents\Erasmus\AVS\Lab5"
image_dir = path + "\pairs"

number_of_images = 49
for i in range(1, number_of_images):
    # Read left image
    img_l = cv2.imread(image_dir + "\left_%02d.png" % i)
    gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)

    # Read right image
    img_r = cv2.imread(r"c:\Users\anton\Documents\Erasmus\AVS\Lab5\pairs\right_%02d.png" % i)
    gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners in the left image
    ret_l, corners_l = cv2.findChessboardCorners(gray_l, (width, height),
                                                  cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

    # Find the chessboard corners in the right image
    ret_r, corners_r = cv2.findChessboardCorners(gray_r, (width, height),
                                                  cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

    if ret_l and ret_r:
        objpoints.append(objp)

        # Improve the location of points (sub-pixel)
        corners_l = cv2.cornerSubPix(gray_l, corners_l, (3, 3), (-1, -1), criteria)
        imgpointsLeft.append(corners_l)

        corners_r = cv2.cornerSubPix(gray_r, corners_r, (3, 3), (-1, -1), criteria)
        imgpointsRight.append(corners_r)

    else:
        print("Chessboard couldn't be detected. Image pair:", i)

# Calibrate left camera
ret_l, K_left, D_left, _, _ = cv2.fisheye.calibrate(
    objpoints, imgpointsLeft, gray_l.shape[::-1], None, None, None, None,
    calibration_flags, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
)

# Calibrate right camera
ret_r, K_right, D_right, _, _ = cv2.fisheye.calibrate(
    objpoints, imgpointsRight, gray_r.shape[::-1], None, None, None, None,
    calibration_flags, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
)

# Load example0 image
example_image = cv2.imread(r"c:\Users\anton\Documents\Erasmus\AVS\Lab5\example\example0.jpg")
height, width, _ = example_image.shape

# Split example image into left and right halves
left_half = example_image[:, :width // 2]
right_half = example_image[:, width // 2:]

# Convert halves to grayscale
gray_left = cv2.cvtColor(left_half, cv2.COLOR_BGR2GRAY)
gray_right = cv2.cvtColor(right_half, cv2.COLOR_BGR2GRAY)

# Compute the rectification maps for the left and right cameras
map1_left, map2_left = cv2.fisheye.initUndistortRectifyMap(
    K_left, D_left, np.eye(3), K_left, (width // 2, height), cv2.CV_16SC2)

map1_right, map2_right = cv2.fisheye.initUndistortRectifyMap(
    K_right, D_right, np.eye(3), K_right, (width // 2, height), cv2.CV_16SC2)

# Rectify the example image halves
undistorted_left = cv2.remap(gray_left, map1_left, map2_left, cv2.INTER_LINEAR)
undistorted_right = cv2.remap(gray_right, map1_right, map2_right, cv2.INTER_LINEAR)

# Compute the disparity maps for the rectified example image halves
stereo_bm_left = cv2.StereoBM_create(numDisparities=16, blockSize=15)
disparity_bm_left = stereo_bm_left.compute(undistorted_left, undistorted_right)

stereo_bm_right = cv2.StereoBM_create(numDisparities=16, blockSize=15)
disparity_bm_right = stereo_bm_right.compute(undistorted_right, undistorted_left)

# Show original images
cv2.imshow("Original Left Half", gray_left)
cv2.imshow("Original Right Half", gray_right)

# Show rectified example image halves
cv2.imshow("Rectified Left Half", undistorted_left)
cv2.imshow("Rectified Right Half", undistorted_right)

# Show disparity maps for the example image halves
cv2.imshow("Disparity Map BM Left Half", disparity_bm_left)
cv2.imshow("Disparity Map BM Right Half", disparity_bm_right)
cv2.waitKey(0)
cv2.destroyAllWindows()
