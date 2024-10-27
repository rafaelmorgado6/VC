import cv2
import numpy as np

# Load the image
image = cv2.imread('/home/rafa/Desktop/ua_computerVision/images/lena.jpg')  # Replace with the path to your image
if image is None:
    print("Error: Image not found.")
    exit()

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply Canny edge detection
edges = cv2.Canny(blurred, 50, 150)  # Adjust these values for better edge detection

# Use the Hough Line Transform to detect lines
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)

# Create a copy of the original image to draw the lines
line_image = np.copy(image)

# If lines were detected, draw them
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Drawing the lines in green

# Display the results
cv2.imshow('Edges', edges)
cv2.imshow('Detected Lines', line_image)

# Wait until any key is pressed, then close the windows
cv2.waitKey(0)
cv2.destroyAllWindows()