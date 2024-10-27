import sys
import numpy as np
import cv2

def printImageFeatures(image):
	
	# Image characteristics
	if len(image.shape) == 2:
		height, width = image.shape
		nchannels = 1;
	else:
		height, width, nchannels = image.shape

	# print some features
	print("Image Height: %d" % height)
	print("Image Width: %d" % width)
	print("Image channels: %d" % nchannels)
	print("Number of elements : %d" % image.size)

image = cv2.imread( '/home/rafa/Desktop/ua_computerVision/images/lena.jpg', cv2.IMREAD_GRAYSCALE );

if  np.shape(image) == (): # Failed Reading
	print("Image file could not be open!")
	exit(-1)

printImageFeatures(image)

cv2.imshow('Orginal', image)

# Sobel Operatot 3 x 3 (técnica de detecção de bordas que aplica derivadas sobre uma imagem)
imageSobel3x3_X = cv2.Sobel(image, cv2.CV_64F, 1, 0, 3)


cv2.namedWindow( "Sobel 3 x 3 - X", cv2.WINDOW_AUTOSIZE )
cv2.imshow( "Sobel 3 x 3 - X", imageSobel3x3_X )
image8bits = np.uint8( np.absolute(imageSobel3x3_X) )
cv2.imshow( "8 bits - Sobel 3 x 3 - X", image8bits )

# Detector de bordas de Canny
edges_1x255 = cv2.Canny(image, 1, 255)
cv2.imshow( "Edges_1x255", edges_1x255 )

# Detector de bordas de Canny
edges_220x225 = cv2.Canny(image, 220, 225)
cv2.imshow( "edges_220x225", edges_220x225 )

# Detector de bordas de Canny
edges_1x128 = cv2.Canny(image, 1, 128)
cv2.imshow( "edges_1x128", edges_1x128 )

cv2.waitKey(0)

# Sobel -> mais simples e rápido, mas menos preciso na deteção de bordas
# Canny ->  mais complexo, requer mais processamento, mas entrega resultados mais robustos e detalhados para detecção de bordas
