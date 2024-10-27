# aula_03_exe_06.py
#
# RGB and Gray-Level Histograms Visualization
#
# Paulo Dias

#import necessary libraries
import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Read the RGB image
image = cv2.imread(r'/home/rafa/Desktop/ua_computerVision/images/deti.bmp', cv2.IMREAD_COLOR)

if image is None:
    print("Image file could not be opened!")
    exit(-1)

# Split the image into its Red, Green, and Blue components
b, g, r = cv2.split(image)

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Display the original RGB image and grayscale image
cv2.imshow("Original RGB Image", image)
cv2.imshow("Gray-Level Image", gray_image)

# Compute histograms for each color channel (Blue, Green, Red)
histSize = 256  # From 0 to 255
histRange = [0, 256]

# Compute histograms for the B, G, R components
hist_b = cv2.calcHist([b], [0], None, [histSize], histRange)
hist_g = cv2.calcHist([g], [0], None, [histSize], histRange)
hist_r = cv2.calcHist([r], [0], None, [histSize], histRange)

# Compute histogram for the grayscale image
hist_gray = cv2.calcHist([gray_image], [0], None, [histSize], histRange)

histImageWidth = 512
histImageHeight = 512
color = 125

histImage  = np.zeros((histImageWidth,histImageHeight,1), np.uint8)
histImageColor = np.zeros((histImageHeight, histImageWidth, 3), np.uint8)  # Color

# Width of each histogram bar
binWidth = int (np.ceil(histImageWidth*1.0 / histSize))

# Normalize histograms
cv2.normalize(hist_b, hist_b, 0, histImageHeight, cv2.NORM_MINMAX)
cv2.normalize(hist_g, hist_g, 0, histImageHeight, cv2.NORM_MINMAX)
cv2.normalize(hist_r, hist_r, 0, histImageHeight, cv2.NORM_MINMAX)
cv2.normalize(hist_gray, hist_gray, 0, histImageHeight, cv2.NORM_MINMAX)

# Draw the bars of the nomrmalized histogram
for i in range (histSize):
    cv2.rectangle(histImage ,  ( i * binWidth , 0 ), ( ( i + 1 ) * binWidth, int(hist_gray[i]) ), (125), -1)
    cv2.line(histImageColor, (i - 1, histImageHeight - int(hist_b[i - 1])), (i, histImageHeight - int(hist_b[i])), (255, 0, 0), 2)
    cv2.line(histImageColor, (i - 1, histImageHeight - int(hist_g[i - 1])), (i, histImageHeight - int(hist_g[i])), (0, 255, 0), 2) 
    cv2.line(histImageColor, (i - 1, histImageHeight - int(hist_r[i - 1])), (i, histImageHeight - int(hist_r[i])), (0, 0, 255), 2)

# ATTENTION : Y coordinate upside down
histImage  = np.flipud(histImage)
#histImageColor = np.flipud(histImageColor)


cv2.imshow('Gray histogram', histImage)
cv2.imshow('Color histogram', histImageColor)


# Wait for key press to close all OpenCV windows
cv2.waitKey(0)
cv2.destroyAllWindows()