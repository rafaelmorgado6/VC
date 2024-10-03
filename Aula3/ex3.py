# Aula_03_ex_03.py
#
# Historam visualization with openCV
#
# Paulo Dias


import numpy as np
import cv2

# Read the image from argv
# image = cv2.imread( sys.argv[1] , cv2.IMREAD_UNCHANGED );
image = cv2.imread( r'/home/rafa/Desktop/VC/Images/ireland-06-06.tif', cv2.IMREAD_UNCHANGED );

if np.shape(image) == ():
	# Failed Reading
	print("Image file could not be open!")
	exit(-1)

# Image characteristics
if len (image.shape) > 2:
	print ("The loaded image is NOT a GRAY-LEVEL image !")
	exit(-1)

# Display the image
cv2.namedWindow("Original Image")
cv2.imshow("Original Image", image)

# print some features
height, width = image.shape
nchannels = 1
print("Image Size: (%d,%d)" % (height, width))
print("Image Type: %d" % nchannels)
print("Number of elements : %d" % image.size)

print("Image Size: (%d,%d)" % (height, width))

# Size
histSize = 256	 # from 0 to 255
# Intensity Range
histRange = [0, 256]

# Compute the histogram
hist_item = cv2.calcHist([image], [0], None, [histSize], histRange)

##########################################
# Drawing with openCV
# Create an image to display the histogram
histImageWidth = 512
histImageHeight = 512
color = 125
histImage = np.zeros((histImageWidth,histImageHeight,1), np.uint8)

# Width of each histogram bar
binWidth = int (np.ceil(histImageWidth*1.0 / histSize))

# Normalize values to [0, histImageHeight]
cv2.normalize(hist_item, hist_item, 0, histImageHeight, cv2.NORM_MINMAX)

# Draw the bars of the nomrmalized histogram
for i in range (histSize):
	cv2.rectangle(histImage,  ( i * binWidth, 0 ), ( ( i + 1 ) * binWidth, int(hist_item[i]) ), (125), -1)

# ATTENTION : Y coordinate upside down
histImage = np.flipud(histImage)

cv2.imshow('colorhist', histImage)
cv2.waitKey(0)


##########################
'''
# Drawing using matplotlib
plt.plot(hist_item,'r')
plt.xlim(histRange)
plt.show()
'''
