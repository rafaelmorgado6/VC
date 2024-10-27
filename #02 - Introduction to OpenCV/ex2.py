# Aula_02_ex_02.py
 #
 # Example of visualization of an image with openCV
 #
 # Paulo Dias

import cv2
import sys

# Lê o nome do arquivo da imagem da linha de comando
image_file = sys.argv[0]

# Read the image
image1 = cv2.imread( r'/home/rafa/Desktop/ua_computerVision/images/deti.bmp', cv2.IMREAD_COLOR )
image2 = cv2.imread( r'/home/rafa/Desktop/ua_computerVision/images/deti.jpg', cv2.IMREAD_COLOR )


if image1 is None or image2 is None:
    print("Uma das imagens não pôde ser aberta.")
    exit(-1)

# Image characteristics
height, width, channels = image1.shape
height1, width1, channels1 = image2.shape

result_image = cv2.subtract(image1, image2)

print("Image1 Size: (%d,%d,%d)" % (height, width, channels))
print("Image1 Type: %s" % image1.dtype)
print("Image2 Size: (%d,%d,%d)" % (height1, width1, channels1))
print("Image2 Type: %s" % image2.dtype)

# Show the image
cv2.imshow('deti.bmp', image1)
cv2.imshow('deti.jpg', image2)
cv2.imshow('Result', result_image)

# Wait
cv2.waitKey( 0 )

# Destroy the window -- might be omitted
cv2.destroyWindow( "Display window" )
