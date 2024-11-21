import numpy as np
import cv2

# Load the calibration parameters from stereoParams.npz
params = np.load("stereoParams.npz")
intrinsics1 = params["intrinsics1"]
distortion1 = params["distortion1"]
intrinsics2 = params["intrinsics2"]
distortion2 = params["distortion2"]

# Load a stereo pair of images (left and right)
imgL = cv2.imread('..//images//left01.jpg')
imgR = cv2.imread('..//images//right01.jpg')

# Undistort the images using the intrinsics and distortion parameters
imgL_undistorted = cv2.undistort(imgL, intrinsics1, distortion1)
imgR_undistorted = cv2.undistort(imgR, intrinsics2, distortion2)

# Display original and undistorted images side by side for comparison
cv2.imshow("Left Original", imgL)
cv2.imshow("Left Undistorted", imgL_undistorted)
cv2.imshow("Right Original", imgR)
cv2.imshow("Right Undistorted", imgR_undistorted)

print("Press any key to close the images.")
cv2.waitKey(0)
cv2.destroyAllWindows()
