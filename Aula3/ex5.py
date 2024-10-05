# Aula_03_ex_03.py
#
# Historam visualization with openCV
#
# Paulo Dias


import numpy as np
import cv2
def main():
	# Read the image from argv
	# image = cv2.imread( sys.argv[1] , cv2.IMREAD_UNCHANGED );
	image = cv2.imread( r'/home/rafa/Desktop/VC/Images/TAC_PULMAO.bmp', cv2.IMREAD_UNCHANGED );

	if np.shape(image) == ():
		# Failed Reading
		print("Image file could not be open!")
		exit(-1)

	# Image characteristics
	if len (image.shape) > 2:
		print ("The loaded image is NOT a GRAY-LEVEL image !")
		exit(-1)

	# Display the image
	cv2.imshow("TAC_PULMAO.bmp", image)

	equalized_image = cv2.equalizeHist(image)

	# Display the contrast-stretched image
	cv2.imshow("Contrast-Stretched Image", equalized_image)

	# Size
	histSize = 256	 # from 0 to 255
	# Intensity Range
	histRange = [0, 256]

	# Compute the histogram
	hist_item = cv2.calcHist([image], [0], None, [histSize], histRange)
	hist_item1 = cv2.calcHist([equalized_image], [0], None, [histSize], histRange)

	##########################################
	# Drawing with openCV
	# Create an image to display the histogram
	histImageWidth = 512
	histImageHeight = 512
	color = 125
	histImage = np.zeros((histImageWidth,histImageHeight,1), np.uint8)
	histImage1 = np.zeros((histImageWidth,histImageHeight,1), np.uint8)

	# Width of each histogram bar
	binWidth = int (np.ceil(histImageWidth*1.0 / histSize))

	# Normalize values to [0, histImageHeight]
	cv2.normalize(hist_item, hist_item, 0, histImageHeight, cv2.NORM_MINMAX)
	cv2.normalize(hist_item1, hist_item1, 0, histImageHeight, cv2.NORM_MINMAX)

	# Draw the bars of the nomrmalized histogram
	for i in range (histSize):
		cv2.rectangle(histImage,  ( i * binWidth, 0 ), ( ( i + 1 ) * binWidth, int(hist_item[i]) ), (125), -1)
		cv2.rectangle(histImage1,  ( i * binWidth, 0 ), ( ( i + 1 ) * binWidth, int(hist_item1[i]) ), (125), -1)

	# ATTENTION : Y coordinate upside down
	histImage = np.flipud(histImage)
	histImage1 = np.flipud(histImage1)

	cv2.imshow('colorhist', histImage)
	cv2.imshow('Equalized', histImage1)

	cv2.waitKey(0)
	cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

