import numpy as np
import cv2

# Load the calibration parameters
params = np.load("stereoParams.npz")
intrinsics1 = params["intrinsics1"]
distortion1 = params["distortion1"]
intrinsics2 = params["intrinsics2"]
distortion2 = params["distortion2"]
F = params["F"]  # Fundamental matrix from stereo calibration

# Load a stereo pair of images (left and right)
imgL = cv2.imread('..//images//left01.jpg')
imgR = cv2.imread('..//images//right01.jpg')

# Undistort the images
imgL_undistorted = cv2.undistort(imgL, intrinsics1, distortion1)
imgR_undistorted = cv2.undistort(imgR, intrinsics2, distortion2)

# Function to handle mouse events and draw epipolar lines
def mouse_handler_left(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Print selected point in the left image
        print(f"Left image point selected: ({x}, {y})")
        
        # Compute epipolar line in the right image
        p = np.array([x, y])
        epilineR = cv2.computeCorrespondEpilines(p.reshape(-1, 1, 2), 1, F)
        epilineR = epilineR.reshape(-1, 3)[0]
        
        # Draw the epipolar line on the right image
        color = np.random.randint(0, 255, 3).tolist()
        x0, y0 = map(int, [0, -epilineR[2] / epilineR[1]])
        x1, y1 = map(int, [imgR_undistorted.shape[1], -(epilineR[2] + epilineR[0] * imgR_undistorted.shape[1]) / epilineR[1]])
        cv2.line(imgR_display, (x0, y0), (x1, y1), color, 2)
        cv2.imshow("Right Image - Undistorted", imgR_display)

def mouse_handler_right(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Print selected point in the right image
        print(f"Right image point selected: ({x}, {y})")
        
        # Compute epipolar line in the left image
        p = np.array([x, y])
        epilineL = cv2.computeCorrespondEpilines(p.reshape(-1, 1, 2), 2, F)
        epilineL = epilineL.reshape(-1, 3)[0]
        
        # Draw the epipolar line on the left image
        color = np.random.randint(0, 255, 3).tolist()
        x0, y0 = map(int, [0, -epilineL[2] / epilineL[1]])
        x1, y1 = map(int, [imgL_undistorted.shape[1], -(epilineL[2] + epilineL[0] * imgL_undistorted.shape[1]) / epilineL[1]])
        cv2.line(imgL_display, (x0, y0), (x1, y1), color, 2)
        cv2.imshow("Left Image - Undistorted", imgL_display)

# Create copies of the undistorted images to draw epipolar lines
imgL_display = imgL_undistorted.copy()
imgR_display = imgR_undistorted.copy()

# Show undistorted images and set mouse callbacks
cv2.imshow("Left Image - Undistorted", imgL_display)
cv2.imshow("Right Image - Undistorted", imgR_display)

cv2.setMouseCallback("Left Image - Undistorted", mouse_handler_left)
cv2.setMouseCallback("Right Image - Undistorted", mouse_handler_right)

print("Click on points in the undistorted images to generate corresponding epipolar lines.")
print("Press any key to exit.")

cv2.waitKey(-1)  # Wait indefinitely until a key is pressed
cv2.destroyAllWindows()