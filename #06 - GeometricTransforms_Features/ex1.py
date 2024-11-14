import numpy as np
import cv2
    
image = cv2.imread(r'/home/rafa/Desktop/VC/images/deti.bmp',  cv2.IMREAD_UNCHANGED)

# Check if images were loaded correctly
if image is None:
    print("Error: One of the images could not be loaded.")

# Obter as dimensões da imagem
rows, cols, channels = image.shape

# Criar a matriz de rotação: (centro, ângulo, escala)
M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 25, 1)  # Rotaciona 25 graus
M[0][2] += -50  # Translação em X
M[1][2] += 100  # Translação em Y

# Aplicar a transformação à imagem
transformed_image = cv2.warpAffine(image, M, (cols, rows))

cv2.imshow("deti.bmp", image)
cv2.imshow("transformed deti.bmp", transformed_image)

# Salvar a imagem transformada
cv2.imwrite('/home/rafa/Desktop/ua_computerVision/images/deti_tf.bmp', transformed_image)

cv2.waitKey(0)
cv2.destroyAllWindows()