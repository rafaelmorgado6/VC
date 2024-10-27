import cv2
import numpy as np
import math

# Initialize global variables to store the selected points
srcPts = []
dstPts = []

# Load the original and transformed images
src = cv2.imread('/home/rafa/Desktop/ua_computerVision/images/deti.bmp')  # Load the original image
dst = cv2.imread('/home/rafa/Desktop/ua_computerVision/images/deti_tf.bmp')  # Load the transformed image

# Check if images are loaded
if src is None or dst is None:
    print("Error loading images. Check the image paths.")
    exit()

# Function to select points from the original image
def select_src(event, x, y, flags, params):
    global srcPts
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(srcPts) < 3:
            srcPts.append((x,y))
            cv2.circle(src, (x, y), 2, (255, 0, 0), 2)
            cv2.putText(src, str(len(srcPts)), (x+10, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
            cv2.imshow("Original", src)

# Function to select points from the transformed image
def select_dst(event, x, y, flags, params):
    global dstPts
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(dstPts) < 3:
            dstPts.append((x,y))
            cv2.circle(dst, (x, y), 2, (0, 255, 0), 2)
            cv2.putText(dst, str(len(dstPts)), (x+10, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
            cv2.imshow("Transformed", dst)

# Display the images and allow point selection
cv2.imshow("Original", src)
cv2.imshow("Transformed", dst)

cv2.setMouseCallback("Original", select_src)
cv2.setMouseCallback("Transformed", select_dst)

# Wait until 3 points are selected in both images
print("Please select 3 corresponding points in both images by clicking on them.")
cv2.waitKey(0)
cv2.destroyAllWindows()

# Ensure exactly 3 points are selected in each image
if len(srcPts) == 3 and len(dstPts) == 3:
    # Convert selected points to numpy arrays
    np_srcPts = np.array(srcPts).astype(np.float32)
    np_dstPts = np.array(dstPts).astype(np.float32)

    # Estimate the affine transformation matrix that transforms srcPts em dstPts
    affine_matrix = cv2.getAffineTransform(np_srcPts, np_dstPts)

    # Print the estimated affine transformation matrix
    print("Affine Transformation Matrix:")
    print(affine_matrix)

    # Warp the original image using the affine transformation
    warp_dst = cv2.warpAffine(src, affine_matrix, (src.shape[1], src.shape[0]))

    # Display the warped image
    cv2.imshow("Warped Image", warp_dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Extract parameters from the affine matrix
    a, c, tx = affine_matrix[0]
    b, d, ty = affine_matrix[1]

    # Compute translation
    t_x = tx
    t_y = ty

    # Compute scaling factors
    s_x = np.sign(a) * math.sqrt(a*2 + b*2)
    s_y = np.sign(d) * math.sqrt(c*2 + d*2)

    # Compute rotation angle
    psi = math.atan2(b, a)

    # Convert rotation angle from radians to degrees
    rotation_angle = np.degrees(psi)

    # Display computed parameters
    print(f"Translation: t_x = {t_x}, t_y = {t_y}")
    print(f"Scale: s_x = {s_x}, s_y = {s_y}")
    print(f"Rotation angle (in degrees): {rotation_angle}")

    # Convert both images to grayscale for subtraction comparison
    warp_dst_gray = cv2.cvtColor(warp_dst, cv2.COLOR_BGR2GRAY)
    dst_gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

    # Compute the absolute difference between the two images
    difference = cv2.absdiff(warp_dst_gray, dst_gray)

    # Display the difference
    cv2.imshow("Difference between Warped and Transformed", difference)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Please select exactly 3 points in each image.")