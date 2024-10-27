# Aula_01_ex_01.py
 #
 # Example of visualization of an image with openCV
 #
 # Paulo Dias

import numpy as np 	# fornece suporte para arrays multidimensionais (matrizes) e uma coleção de funções matemáticas e lógicas para operar com esses arrays
import cv2		# módulo principal da biblioteca OpenCV
import sys			#  módulo principal da biblioteca OpenCV

# Read the image  r'' serve para nao causar problemas com "/"
# cv2.IMREAD_UNCHANGED -> carrega a imagem exatamente como ela é, sem modificar suas propriedades
image = cv2.imread(r'/home/rafa/Desktop/ua_computerVision/images/lena.jpg', cv2.IMREAD_UNCHANGED)


if  np.shape(image) == ():	# função da biblioteca np(numpy)
	# Failed Reading
	print("Image file could not be open")
	exit(-1)
# Exit(0) -> indica que o programa foi terminado sem erros
# Exit(-1) -> indica que o programa foi terminado devido a um erro

# Image characteristics
height, width = image.shape	# image.shape retorna as dimensões da imagem em forma de uma tupla

# (%d,%d)" % (height, width)-> faz com que height e width substituiam %d,%d na string, %d(int)
print("Image Size: (%d,%d)" % (height, width))
# %s" % (image.dtype)-> o tipo de dados do array substitui %s(string)
print("Image Type: %s" % (image.dtype))	#image.dtype retorna o tipo de dados do array que representa a imagem
										# uint8: valores dos pixeis vão de 0 a 255
										# float 32: valores dos pixeis são representados como numeros de ponto flutunte

# Create a vsiualization window (optional)
# CV_WINDOW_AUTOSIZE : window size will depend on image size
cv2.namedWindow( "Display window", cv2.WINDOW_AUTOSIZE )

# Show the image
cv2.imshow( "Display window", image )

# 0 -> deifne o tempo em milissegundos que a função deve esperar pela entrada do teclado
# sem este comando a imagem fecha logo a seguir a ser chamada
cv2.waitKey( 0 )

# Destroy the window -- might be omitted
cv2.destroyWindow( "Display window" )

