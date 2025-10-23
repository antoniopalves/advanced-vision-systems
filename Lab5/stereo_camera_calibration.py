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

img_width = 640
img_height = 480
image_size = (img_width, img_height)

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

N_OK = len(objpoints)
K_left = np.zeros((3, 3))
D_left = np.zeros((4, 1))
K_right = np.zeros((3, 3))
D_right = np.zeros((4, 1))
rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(N_OK)]
tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(N_OK)]

# Calibrate left camera
ret_l, K_left, D_left, _, _ = cv2.fisheye.calibrate(
    objpoints, imgpointsLeft, image_size, K_left, D_left, rvecs, tvecs,
    calibration_flags, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
)

# Calibrate right camera
ret_r, K_right, D_right, _, _ = cv2.fisheye.calibrate(
    objpoints, imgpointsRight, image_size, K_right, D_right, rvecs, tvecs,
    calibration_flags, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
)

# Stereo calibration
objpoints = np.asarray(objpoints, dtype=np.float64)
imgpointsLeft = np.asarray(imgpointsLeft, dtype=np.float64)
imgpointsRight = np.asarray(imgpointsRight, dtype=np.float64)

(RMS, _, _, _, _, rotationMatrix, translationVector) = cv2.fisheye.stereoCalibrate(
    objpoints, imgpointsLeft, imgpointsRight,
    K_left, D_left,
    K_right, D_right,
    image_size, None, None,
    cv2.CALIB_FIX_INTRINSIC,
    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
)


R2 = np.zeros([3, 3])
P1 = np.zeros([3, 4])
P2 = np.zeros([3, 4])
Q = np.zeros([4, 4])

# Rectify calibration results
(leftRectification, rightRectification, leftProjection, rightProjection, disparityToDepthMap) = cv2.fisheye.stereoRectify(
    K_left, D_left,
    K_right, D_right,
    image_size,
    rotationMatrix, translationVector,
    0, R2, P1, P2, Q,
    cv2.CALIB_ZERO_DISPARITY, (0, 0), 0, 0
)

map1_left, map2_left = cv2.fisheye.initUndistortRectifyMap(
    K_left, D_left, leftRectification,
    leftProjection, image_size, cv2.CV_16SC2)

map1_right, map2_right = cv2.fisheye.initUndistortRectifyMap(
    K_right, D_right, rightRectification,
    rightProjection, image_size, cv2.CV_16SC2)

dst_L = cv2.remap(img_l, map1_left, map2_left, cv2.INTER_LINEAR)
dst_R = cv2.remap(img_r, map1_right, map2_right, cv2.INTER_LINEAR)

N, XX, YY = dst_L.shape[::-1]  # RGB image size

visRectify = np.zeros((YY, XX * 2, N), np.uint8)  # Create a new image with a new size (height, 2*width)
visRectify[:, 0:XX:, :] = dst_L  # Left image assignment
visRectify[:, XX:XX * 2:, :] = dst_R  # Right image assignment

# Draw horizontal lines
for y in range(0, YY, 10):
    cv2.line(visRectify, (0, y), (XX * 2, y), (255, 0, 0))

cv2.imshow('Stereo Calibration and Rectification', visRectify)  # Display image with lines
cv2.waitKey(0)
cv2.destroyAllWindows()
