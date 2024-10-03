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


def mouse_handler(event, x, y, flags, params):
    if event == cv2.EVENT_RBUTTONDOWN:
        print("right click")
        cv2.circle(image1, (x,y), 10, (0, 255, 0), -1) # -1 -> circulo preenchido
        cv2.imshow('deti.bmp', image1)

cv2.namedWindow('deti.bmp') # janela para associar à função setMouseCallback
cv2.setMouseCallback("deti.bmp", mouse_handler) #Associa a função mouse_handler com a janela 'deti.bmp'

# Show the image
cv2.imshow('deti.bmp', image1)

# Wait
cv2.waitKey( 0 )

# Destroy the window -- might be omitted
cv2.destroyWindow( "Display window" )
