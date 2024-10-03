# Aula_02_ex_03.py
 #
 # Example of visualization of an image with openCV
 #
 # Paulo Dias

import cv2
import sys

# Lê o nome do arquivo da imagem da linha de comando
image_file = sys.argv[0]

# Read the image
image1 = cv2.imread( r'C:\Users\Rafa\PycharmProjects\VC\deti.bmp', cv2.IMREAD_COLOR )

if image1 is None:
    print("A imagen não pôde ser aberta.")
    exit(-1)

# Image characteristics
height, width, channels = image1.shape

print("Image1 Size: (%d,%d,%d)" % (height, width, channels))
print("Image1 Type: %s" % image1.dtype)

# Colors -> COLOR_RGB2HLS(53), COLOR_RGB2XYZ(33), COLOR_RGB2HSV(41)
grey_image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY) # code -> 7

# Show the image
cv2.imshow('grey deti.bmp', grey_image1)

# Wait
cv2.waitKey( 0 )

# Destroy the window -- might be omitted
cv2.destroyWindow( "Display window" )
