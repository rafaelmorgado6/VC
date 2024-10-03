 # Aula_02_ex_01.py
 #
 # Example of visualization of an image with openCV
 #
 # Paulo Dias

#import
import numpy as np
import cv2
import sys

# Lê o nome do arquivo da imagem da linha de comando
image_file = sys.argv[1]

# Read the image
image = cv2.imread(image_file, cv2.IMREAD_COLOR)

if  image is None:	# np.shape(image) retorna a forma do array
	# Failed Reading
	print("Image file could not be open")
	exit(-1)

image_copy = image.copy()

# Image characteristics
height, width, channels = image.shape

for x in range(height):	# x percorre height
	for y in range (width):	# y percorre width
		if np.any(image[x, y] < 128): # vê se any dos x/y < 120(intensidade)
			image[x, y] = [0, 0, 0] # altera para preto

print("Image Size: (%d,%d,%d)" % (height, width, channels))
print("Image Type: %s" % image_copy.dtype)


# Create a vsiualization window (optional)
# CV_WINDOW_AUTOSIZE : window size will depend on image size
#cv2.namedWindow( "Display window", cv2.WINDOW_AUTOSIZE )

# Show the image
cv2.imshow('image', image)
cv2.imshow('image_copy', image_copy)

# Wait
cv2.waitKey( 0 )

# Destroy the window -- might be omitted
cv2.destroyWindow( "Display window" )
