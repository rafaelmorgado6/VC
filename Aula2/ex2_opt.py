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
image1 = cv2.imread( r'/home/rafa/Desktop/VC/Images/Orchid.bmp', cv2.IMREAD_COLOR )

image_90 = cv2.imread(r'/home/rafa/Desktop/VC/Images/Orchid_90.jpeg')
image_50 = cv2.imread(r'/home/rafa/Desktop/VC/Images/Orchid_50.jpeg')
image_20 = cv2.imread(r'/home/rafa/Desktop/VC/Images/Orchid_20.jpeg')

if image1 is None or image_90 is None or image_50 is None or image_20 is None:
    print("Uma ou mais imagens não puderam ser abertas.")
    exit(-1)

if image1.shape != image_90.shape or image1.shape != image_50.shape or image1.shape != image_20.shape:
    print("As imagens têm tamanhos diferentes!")
    exit(-1)

#if image1.shape != image_90.shape or image1.shape != image_50.shape or image1.shape != image_20.shape:
#    print("As imagens têm tamanhos diferentes!")
#    exit()

result_image90 = cv2.subtract(image1, image_90)
result_image50 = cv2.subtract(image1, image_50)
result_image20 = cv2.subtract(image1, image_20)


# Show the image
# Colors -> COLOR_RGB2HLS(53), COLOR_RGB2XYZ(33), COLOR_RGB2HSV(41)
cv2.imshow('90% compression', result_image90)
cv2.imshow('50% compression', result_image50)
cv2.imshow('20% compression', result_image20)

# Wait
cv2.waitKey( 0 )

# Destroy the window -- might be omitted
cv2.destroyWindow( "Display window" )

# 90% -> Os valores dos pixels resultantes estão estar próximos
# de zero(pretos), indicando que a compressão não
# afetou significativamente a qualidade da imagem.

# 50% ->  Alguns detalhes e texturas começam a ser perdidos,
#  o que indica as partes da imagem que foram alteradas ou
#  distorcidas devido à compressão.

# 20% -> Apresenta áreas mais claras, indicando grandes perdas
# de detalhes e qualidade, machas são visiveis devido aos
# detalhes que foram eliminados ou alterados devido à alta compressão.
