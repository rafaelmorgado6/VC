import cv2
import numpy as np

# Load the original and transformed images
src = cv2.imread('/home/rafa/Desktop/VC/images/deti.bmp')  # Load the original image
dst = cv2.imread('/home/rafa/Desktop/VC/images/deti_tf.bmp')  # Load the transformed image

# Check if images are loaded
if src is None or dst is None:
    print("Error loading images. Check the image paths.")
    exit()

# Initiate SIFT detector
sift = cv2.SIFT_create()

# Find the keypoints (borders, corners, different textures) and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(src, None)
kp2, des2 = sift.detectAndCompute(dst, None)

# Draw the keypoints on the images
src_keypoints = cv2.drawKeypoints(src, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
dst_keypoints = cv2.drawKeypoints(dst, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Display the keypoints in each image
cv2.imshow("Keypoints in Original Image", src_keypoints)
cv2.imshow("Keypoints in Transformed Image", dst_keypoints)

# Wait until a key is pressed and close the windows
cv2.waitKey(0)
cv2.destroyAllWindows()