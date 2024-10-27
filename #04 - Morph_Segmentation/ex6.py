import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread( r'/home/rafa/Desktop/ua_computerVision/images/lena.jpg', cv2.IMREAD_UNCHANGED )

if  np.shape(image) == ():
	# Failed Reading
	print("Image file could not be open!")
	exit(-1)
	
cv2.imshow(' Image', image)    

# Definir o pixel-semente e os parâmetros de flood fill
seedPoint = (315, 228)  # Coordenadas do pixel-semente
newVal = 255  # Novo valor a ser atribuído aos pixels segmentados
loDiff = 5  # Diferença inferior permitida na intensidade
upDiff = 5  # Diferença superior permitida na intensidade
	
# Criar a máscara necessária para a função floodFill
# A máscara precisa ser 2 pixels maior em cada dimensão que a imagem
height, width = image.shape[:2]
mask = np.zeros((height + 2, width + 2), np.uint8)

retval = cv2.floodFill(image, mask, seedPoint, newVal, loDiff, upDiff)

cv2.imshow('Segmented Image', image)

cv2.waitKey(0)
cv2.destroyAllWindows()