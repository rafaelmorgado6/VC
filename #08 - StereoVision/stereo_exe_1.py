# stereo_exe_1.py
# Stereo Chessboard Calibration
#
# Paulo Dias (Adaptado)

import numpy as np
import cv2
import glob

# Board Size
board_h = 9
board_w = 6

# 3D points in the real world
objp = np.zeros((board_w * board_h, 3), np.float32)
objp[:, :2] = np.mgrid[0:board_w, 0:board_h].T.reshape(-1, 2)

# Arrays to store object points and image points for all the stereo images.
objpoints = []     # 3D points in real world space
left_corners = []  # 2D points in left image plane
right_corners = [] # 2D points in right image plane

def find_and_display_chessboard(img, board_size):
    """Find and display chessboard corners in a given image."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, board_size, None)
    
    if ret:
        cv2.drawChessboardCorners(img, board_size, corners, ret)
        cv2.imshow('Chessboard Detection', img)
        cv2.waitKey(500)
        
    return ret, corners

# Get sorted lists of left and right images
left_images = sorted(glob.glob('..//images//left*.jpg'))
right_images = sorted(glob.glob('..//images//right*.jpg'))

# Check if we have equal number of left and right images
if len(left_images) != len(right_images):
    print("Número desigual de imagens para esquerda e direita.")
    exit()

# Process each pair of images
for left_fname, right_fname in zip(left_images, right_images):
    left_img = cv2.imread(left_fname)
    right_img = cv2.imread(right_fname)

    # Detect corners in both left and right images
    ret_left, corners_left = find_and_display_chessboard(left_img, (board_w, board_h))
    ret_right, corners_right = find_and_display_chessboard(right_img, (board_w, board_h))

    # If corners are detected in both images, add to lists
    if ret_left and ret_right:
        objpoints.append(objp)
        left_corners.append(corners_left)
        right_corners.append(corners_right)

cv2.destroyAllWindows()

# Agora `left_corners`, `right_corners` e `objpoints` têm os pontos necessários para a calibração estéreo.
# Estes podem ser usados com `cv2.stereoCalibrate()` para calcular as matrizes de calibração.

# (Opcional: Calibração estéreo e salvamento das matrizes de calibração)
